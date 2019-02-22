import torch
import torch.nn.functional as F

from .inv import inv


class FisherBlock():

    def __init__(self,damping=0.01,cov_ema_decay=0.99,bias=True,pi_type='trace_norm'):
        self.cov_ema_decay = cov_ema_decay
        self.bias = bias
        self.damping = damping
        self.covs_ema = None
        self.pi_type = pi_type

    
    def compute_covs(self,kfac_buf):
        return (self.compute_A(kfac_buf[0]),self.compute_G(kfac_buf[1]))
    
    def compute_G(self,grad_output_data):
        raise NotImplementedError

    def compute_A(self,input_data):
        raise NotImplementedError

    def update_covs_ema(self,covs):
        if self.covs_ema is None:
            self.covs_ema = covs
        else:
            alpha = self.cov_ema_decay
            A,G = covs
            A_ema = A.mul(alpha).add(1-alpha,self.covs_ema[0])
            G_ema = G.mul(alpha).add(1-alpha,self.covs_ema[1])
            self.covs_ema = A_ema,G_ema
    
    def compute_pi_tracenorm(self,covs):
        A,G = covs
        A_size,G_size = A.shape[0],G.shape[0]

        return torch.sqrt((A.trace()/(A_size))/(G.trace()/(G_size)))

    def compute_damped_covs(self,covs):
        if self.pi_type == 'trace_norm':
            pi = self.compute_pi_tracenorm(covs)
        else:
            pi = 1
        r = self.damping**0.5
        pi = float(pi)
        A,G = covs
        A_damping = torch.diag(torch.ones(A.shape[0],device = A.device))        
        G_damping = torch.diag(torch.ones(G.shape[0],device = G.device))
        A.add_(r*pi,A_damping)
        G.add_(r/pi,G_damping)

        return A,G


    def compute_kfgrad(self,tp,kfac_buf):
        raise NotImplementedError 
    
    def __call__(self,tp,kfac_buf):
        return self.compute_kfgrad(tp,kfac_buf)



class LinearFB(FisherBlock):


    def compute_G(self,grad_ouput_data):
        batch_size = grad_ouput_data.shape[0]
        return grad_ouput_data.transpose(0,1).mm(grad_ouput_data).mul(1/batch_size)

    def compute_A(self,input_data):
        batch_size = input_data.shape[0]
        if self.bias:
            input_data = torch.cat((input_data,torch.ones((batch_size,1),device=input_data.device)),1)
        return input_data.transpose(0,1).mm(input_data).mul(1/batch_size)
    
    def compute_kfgrad(self,tp,kfac_buf):
        A,G = self.compute_covs(kfac_buf)

        if self.cov_ema_decay != 0:
            self.update_covs_ema((A,G))
            A,G = self.covs_ema

        A,G = self.compute_damped_covs((A,G))

        A_inv,G_inv = inv(A),inv(G)

        param_grad = tp[0].grad
        if self.bias:
            param_grad = torch.cat((param_grad,tp[1].grad.view(-1,1)),1)
            kfgrad = G_inv.mm(param_grad).mm(A_inv)
            return kfgrad[:,0:-1],kfgrad[:,-1]
        else:
            kfgrad = G_inv.mm(param_grad).mm(A_inv)
            return (kfgrad,)


class Conv2dFB(FisherBlock):

    def __init__(self,kernel_size,stride,padding,dilation,damping=0.01,cov_ema_decay=0.99,bias=True,pi_type='trace_norm'):
        super(Conv2dFB,self).__init__(damping,cov_ema_decay,bias,pi_type)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def compute_G(self,grad_output_data):
        batch_size,c,h,w = grad_output_data.shape
        m = grad_output_data.transpose(0,1).reshape(c,-1)
        
        return m.mm(m.transpose(0,1)).mul(1/(batch_size*h*w))

    def compute_A(self,input_data):
        input_data2d = F.unfold(input_data,kernel_size=self.kernel_size,stride=self.stride,padding=self.padding,dilation=self.dilation) 
        batch_size,a,_ = input_data2d.shape
        m = input_data2d.transpose(0,1).reshape(a,-1)
        a,b = m.shape
        if self.bias:
            m = torch.cat((m,torch.ones((1,b),device=input_data.device)),0)
        

        return m.mm(m.transpose(0,1)).mul(1/batch_size)
    
    def compute_kfgrad(self,tp,kfac_buf):
        A,G = self.compute_covs(kfac_buf)
        if self.cov_ema_decay != 0:
            self.update_covs_ema((A,G))
            A,G = self.covs_ema

        A,G = self.compute_damped_covs((A,G))
        
        A_inv,G_inv = inv(A),inv(G) 
        
        param_grad = tp[0].grad
        oc,ic,h,w = param_grad.shape
        param_grad2d = param_grad.reshape(oc,-1)
        if self.bias:
            param_grad2d = torch.cat((param_grad2d,tp[1].grad.view(-1,1)),1)
            kfgrad2d = G_inv.mm(param_grad2d).mm(A_inv)
            return kfgrad2d[:,0:-1].reshape(oc,ic,h,w),kfgrad2d[:,-1]
        else:
            kfgrad2d = G_inv.mm(param_grad2d).mm(A_inv)
            return (kfgrad2d.reshape(oc,ic,h,w),)
