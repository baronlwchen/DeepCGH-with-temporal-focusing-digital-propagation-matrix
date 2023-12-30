# -*- coding: utf-8 -*-
# %%
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
from skimage.draw import circle, line_aa
import numpy as np
import numpy.matlib
import scipy.io as scio

import math
import cmath
import matplotlib.pyplot as plt
import gc

class GS3D(object):
    '''
    Class for the GS algorithm.
    Inputs:
        batch_size   int, determines the batch size of the prediction
        num_iter   int, determines the number of iterations of the GS algorithm
        input_shape   tuple of shape (height, width)
    Returns:
        Instance of the object
    '''
    def __init__(self,
                 data_params,
                 model_params):
        self.shape = data_params['shape']
        self.plane_distance = model_params['plane_distance']
        self.wavelength = model_params['wavelength']
        self.ps = model_params['pixel_size']
        self.model = model_params
        self.f=model_params['focal_length']
        self.n=model_params['refraction_index']
        self.zs = np.linspace(0-(self.shape[-1]//2),self.shape[-1]//2,self.shape[-1])*self.plane_distance
        self.Hs = self.__get_H(self.zs, self.shape, self.wavelength, self.ps,self.f,self.n)
        
    def __get_H(self, zs, shape, lambda_, ps,f,n):
        Hs = []
        k=np.pi*2/lambda_*n
        for z in zs:
            x, y = np.meshgrid(np.linspace(-shape[1]//2+1, shape[1]//2, shape[1]),
                                   np.linspace(-shape[0]//2+1, shape[0]//2, shape[0]))#以中心為(0,0)的像素x,y座標
            #ps_=lambda_*f/ps/shape[0]
            #delta_f=1/ps_/shape[0]
            fx = x*ps/(lambda_*f)
            fy = y*ps/(lambda_*f)
                
            kz=np.sqrt(k**2-(2*np.pi*fx)**2-(2*np.pi*fy)**2)
            kz=np.nan_to_num(kz)
            exp = np.exp(1j * kz.real * z ) #fresnel diffraction transfer function?
            #exp = np.exp(-1j * np.pi * lambda_ * z * (fx**2 + fy**2))
            Hs.append(exp.astype(np.complex64))
        '''
        Hs = []
        #radius=shape[0]//4
        #location=(shape[0]//2+1,shape[1]//2+1)
        k=np.pi*2/lambda_*n
        for z in zs:
            x, y = np.meshgrid(np.linspace(-shape[1]//2+1, shape[1]//2, shape[1]),
                                   np.linspace(-shape[0]//2+1, shape[0]//2, shape[0]))#以中心為(0,0)的像素x,y座標
            #ps_=lambda_*f/ps/shape[0]
            fx = x*ps/lambda_/f
            fy = y*ps/lambda_/f
                
            kz=np.sqrt(k**2-(2*np.pi*fx)**2-(2*np.pi*fy)**2)
            exp = np.exp(1j * kz.real * z ) #fresnel diffraction transfer function?
            #exp = np.exp(-1j * np.pi * lambda_ * z * (fx**2 + fy**2))
            Hs.append(exp.astype(np.complex64))
        '''
        return Hs

    def __propagate(self, cf, H):
        return np.fft.ifft2(np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(cf))*H))

    def __forward(self, cf_slm, Hs, As):
        new_Z = []
        z0 = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(cf_slm)))
        for H, A in zip(Hs, As):
            if type(H)!=int:
                new_Z.append(A*np.exp(1j*np.angle(self.__propagate(z0, H))))
            else:
                new_Z.append(A*np.exp(1j*np.angle(z0)))
        return new_Z

    def __backward(self, Zs, Hs):
        slm_cfs = []
        for Z, H in zip(Zs, Hs[::-1]):
            if type(H)!=int:
                slm_cfs.append(np.fft.ifft2(np.fft.ifftshift(self.__propagate(Z, H))))
            else:
                slm_cfs.append(np.fft.ifft2(np.fft.ifftshift(Z)))
        cf_slm = np.exp(1j*np.angle(np.sum(np.array(slm_cfs), axis=0)))
        return cf_slm

    def get_phase(self, As, K):
        As = np.transpose(As, axes=(2, 0, 1))
        cf_slm = np.exp(1j * np.random.rand(*As.shape[1:]))
        for i in tqdm(range(K)):
            new_Zs = self.__forward(cf_slm, self.Hs, As)
            cf_slm = self.__backward(new_Zs, self.Hs)
        return np.angle(cf_slm)




def gs2d(img, K):
    phi = np.random.rand(*list(img.shape)).astype(np.float32)
    while K:
        img_cf = img * np.exp(1.j * phi)
        slm_cf = np.fft.ifft2(np.fft.ifftshift(img_cf))
        slm_phi = np.angle(slm_cf)
        slm_cf = 1 * np.exp(1j * slm_phi)
        img_cf = np.fft.fftshift(np.fft.fft2(slm_cf))
        phi = np.angle(img_cf)
        K -= 1
    return slm_phi

def dot2circle(radius,y_true):
    circle_=np.zeros([radius+1,radius+1],dtype=np.float32)
    location=radius/2
    rr, cc = circle(location, location, radius/2, shape=circle_.shape)
    circle_[rr, cc] = 1
    filters=tf.constant(circle_)
    filters=tf.tile(filters[:,:,tf.newaxis,tf.newaxis],[1,1,3,1])
    result=tf.nn.depthwise_conv2d(y_true,filters,strides=[1,1,1,1],padding='SAME')
    print("shape=",result.shape.as_list())
    return result        

def display_results(imgs, phases, recons, t):
    assert imgs.ndim == 4 and phases.ndim == 4 and recons.ndim == 4, "Dimensions don't match"
    #imgs=dot2circle(6,imgs)
    for img, phase, recon in zip(imgs, phases, recons):
        if img.shape[-1] == 1:
            fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True, sharex=True)
            
            axs[0].imshow(np.squeeze(img), cmap='gray')
            axs[0].set_title('Target')
            axs[1].imshow(np.squeeze(phase), cmap='gray')
            axs[1].set_title('SLM Phase')
            axs[2].imshow(np.squeeze(recon), cmap='gray')
            axs[2].set_title('Simulation')
            scio.savemat('img.mat', {'target':img})
            print(type(img))
            recon=np.array(recon)
            scio.savemat('reconstructed.mat', {'recon':recon})
            print(type(recon))
            scio.savemat('SLM_phase.mat', {'SLM_phase': phase})
            print(type(phase))
        else:
            fig, axs = plt.subplots(2, img.shape[-1] + 1, figsize = (3 * (img.shape[-1] + 1), 6), sharey = True, sharex = True)
            axs[0, -1].imshow(np.squeeze(phase))
            axs[0, -1].set_title('SLM Phase')
            for i in range(img.shape[-1]):
                axs[0, i].imshow(img[:, :, i], cmap='gray')
                axs[0, i].set_title('Target @ Z{}'.format(str(i)))
                axs[1, i].imshow(recon[:, :, i], cmap='gray')
                axs[1, i].set_title('Reconstructed')
            scio.savemat('img.mat', {'target':img})
            print(type(img))
            recon=np.array(recon)
            scio.savemat('reconstructed.mat', {'recon':recon})
            print(type(recon))
            scio.savemat('SLM_phase.mat', {'SLM_phase': phase})
            print(type(phase))
            print("x")
        fig.suptitle('Inference time was {:.2f}ms'.format(t*1000), fontsize=16)

def get_propagate(data, model):
    shape = data['shape']
    plane_distance=model['plane_distance']
    zs = np.linspace(0-(shape[-1]//2),shape[-1]//2,shape[-1])*plane_distance
    lambda_ = model['wavelength']
    ps = model['pixel_size']
    f=model['focal_length']
    n=model['refraction_index']
    
    def apply_Gaussian_on_phase(shape):
            IMAGE_WIDTH = shape[0]
            IMAGE_HEIGHT = shape[1]
 
            center_x = IMAGE_WIDTH/2
            center_y = IMAGE_HEIGHT/2
 
            R = np.sqrt(center_x**2 + center_y**2)
 
            Gauss_map = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))
            '''
            # 利用 for 循环 实现
            for i in range(IMAGE_HEIGHT):
                for j in range(IMAGE_WIDTH):
                    dis = np.sqrt((i-center_y)**2+(j-center_x)**2)
                    Gauss_map[i, j] = np.exp(-0.5*dis/R)
            '''
            # 直接利用矩阵运算实现
 
            mask_x = np.matlib.repmat(center_x, IMAGE_HEIGHT, IMAGE_WIDTH)
            mask_y = np.matlib.repmat(center_y, IMAGE_HEIGHT, IMAGE_WIDTH)
 
            x1 = np.arange(IMAGE_WIDTH)
            x_map = np.matlib.repmat(x1, IMAGE_HEIGHT, 1)
 
            y1 = np.arange(IMAGE_HEIGHT)
            y_map = np.matlib.repmat(y1, IMAGE_WIDTH, 1)
            y_map = np.transpose(y_map)
 
            Gauss_map = np.sqrt((x_map-mask_x)**2+(y_map-mask_y)**2)
 
            Gauss_map = np.exp(-0.5*Gauss_map/R)
            
            Gauss=Gauss_map.astype(np.complex64)
            
            return Gauss
        
    def __get_H(zs, shape, lambda_, ps,f,n):
        Hs = []
        #radius=shape[0]//4
        #location=(shape[0]//2+1,shape[1]//2+1)
        k=np.pi*2/lambda_*n
        for z in zs:
            x, y = np.meshgrid(np.linspace(-shape[1]//2+1, shape[1]//2, shape[1]),
                                   np.linspace(-shape[0]//2+1, shape[0]//2, shape[0]))#以中心為(0,0)的像素x,y座標
            #ps_=lambda_*f/ps/shape[0]
            #delta_f=1/ps_/shape[0]
            fx = x*ps/(lambda_*f)
            fy = y*ps/(lambda_*f)
                
            kz=np.sqrt(k**2-(2*np.pi*fx)**2-(2*np.pi*fy)**2)
            kz=np.nan_to_num(kz)
            exp = np.exp(1j * kz.real * z ) #fresnel diffraction transfer function?
            #exp = np.exp(-1j * np.pi * lambda_ * z * (fx**2 + fy**2))
            Hs.append(exp.astype(np.complex64))
        return Hs

    def __prop__(cf_slm, H):
        #print("y")
        #fft=tf.signal.fftshift(tf.signal.fft2d(cf_slm),axes=[1,2])
        #A=tf.signal.fftshift(tf.signal.fft2d(fft),axes=[1,2])
        
        H = tf.broadcast_to(tf.expand_dims(H, axis=0), tf.shape(cf_slm))
        cf_slm *=H
        complex_amp_focal_plane=tf.signal.fftshift(tf.signal.fft2d(cf_slm),axes=[1,2])
        img=tf.cast(tf.expand_dims(complex_amp_focal_plane, axis=-1), dtype=tf.dtypes.complex64)                         
        #cf_slm *= tf.signal.fftshift(H, axes = [1, 2])
        #A*=H
        #U=tf.signal.ifft2d(tf.signal.fftshift(A,axes=[1,2]))
        #fft = tf.signal.ifftshift(tf.signal.fft2d(tf.signal.fftshift(cf_slm, axes = [1, 2])), axes = [1, 2])
            
        #img=tf.expand_dims(U, axis=-1)
        #img=tf.cast(tf.expand_dims(fft, axis=-1), dtype=tf.dtypes.complex64)
        del complex_amp_focal_plane
        gc.collect()
        return img

    def __phi_slm(phi_slm):
        i_phi_slm = tf.dtypes.complex(np.float32(0.), tf.squeeze(phi_slm, axis=-1))
        return tf.math.exp(i_phi_slm)

    Hs = __get_H(zs, shape, lambda_, ps,f,n)
    
    def circle_mask(shape):
        radius=485
        location=(shape[0]//2+1,shape[1]//2+1)
        img = np.zeros(shape[:-1], dtype=np.float32)
        rr, cc = circle(location[0], location[1], radius, shape=img.shape)
        img[rr, cc] = 1
        #plt.plot(rr,cc)
        return img
    
    
    def TF_reconstruction(temporal_focusing):
           
        print("temporal_focusing=",temporal_focusing.shape.as_list())
            
        fft_num=1024
        single_layer=[]
        for i in range(temporal_focusing.shape[1]):
            frequency_component=temporal_focusing[:,i,:,:]
                #print("frequency_component=",frequency_component.shape.as_list())
            frequency_component_=tf.expand_dims(frequency_component,axis=1)
                #print("frequency_component_=",frequency_component_.shape.as_list())
            padding_=tf.constant([[0,0],[0,0],[0,0],[0,fft_num-temporal_focusing.shape[-1]]])
            result=tf.pad(frequency_component_,padding_,"CONSTANT",constant_values=0.)
                #print("after_padding",result.shape.as_list())
            result_=tf.reduce_sum(tf.pow(tf.abs(tf.signal.fftshift(tf.signal.fft(result),axes=(3))),4),axis=-1,keepdims=True)
                #print("timedomain=",result_.shape.as_list())
            single_layer.append(result_)
        imgTF=tf.cast(tf.concat(values=single_layer,axis=1),dtype=tf.dtypes.float32)
        #imgTF_ = (imgTF - tf.reshape(tf.reduce_min(imgTF, axis=(1, 2, 3)), (-1, 1, 1, 1))) / (tf.reshape(tf.reduce_max(imgTF,axis=(1,2,3)), (-1, 1, 1, 1)) - tf.reshape(tf.reduce_min(imgTF, axis=(1, 2, 3)), (-1, 1, 1, 1)))
        imgTF_ = imgTF
        print("imgTF=",imgTF.shape.as_list())
            
            #imgTF_ = (imgTF - tf.reshape(tf.reduce_min(imgTF, axis=(1, 2, 3)), (-1, 1, 1, 1))) / (tf.reshape(tf.reduce_max(imgTF,axis=(1,2,3)), (-1, 1, 1, 1)) - tf.reshape(tf.reduce_min(imgTF, axis=(1, 2, 3)), (-1, 1, 1, 1)))
            #del imgTF
        del result
        del single_layer
        del temporal_focusing
        del frequency_component
        gc.collect()
        return imgTF_
        
    def amplitude_SLM(shape,centralwavelength,pixelsize_SLM):
        shape_=(shape[0]*2,shape[1]*2,shape[-1])
        fL_1=250e-3
        fL_2=150e-3
        beamsize=4.24e-3*fL_2/fL_1
        C=3e8
        pulse_duration=250e-15
        groove=1/600*1e-3;
        bandwidth_FWHM=6.84e-9#11.4e-9#0.441/pulse_duration/C*np.power(centralwavelength,2)
        bandwidth_FWTM=np.sqrt(np.log(10)/np.log(2))*bandwidth_FWHM
        sigma=bandwidth_FWHM/(2*np.sqrt(2*np.log(2)))
        #bandwidth=3*sigma#全寬等於3個sigma(99.1%)
        incident_angle=np.arcsin(centralwavelength/groove)#因為要正打出來，theta_m=0
        #beamsize_x=beamsize/np.cos(incident_angle)

        #lens setup
        #fL_1=200e-3
        #fL_2=75e-3
        FL_fourierlens=750e-3#*fL_1/fL_2 #750e-3*0.75##(focal_length)
        #objective
        n_s=1.333#水鏡
        objective_FL=165/40*1e-3

        spotsize=beamsize*objective_FL/FL_fourierlens

        N_slits=beamsize/groove#佔了grating多少條
        wavelength_increment=centralwavelength/(1*N_slits)
        deltaf=C/centralwavelength**2*wavelength_increment

        totalwavelength=np.arange(centralwavelength-bandwidth_FWHM,centralwavelength+bandwidth_FWHM,wavelength_increment)


        # %%
        mean=np.mean(totalwavelength)

        wavelength_distribution=np.exp(-1*((totalwavelength-mean)**2)/(2*(sigma**2)))/(np.sqrt(2*np.pi) * sigma)
        wavelength_distribution=(wavelength_distribution)/(max(wavelength_distribution))
        plt.figure
        plt.plot(totalwavelength*1e9,wavelength_distribution,marker="o")
        plt.title('amplitude distribution')
        plt.xlabel('Wavelength(nm)')
        plt.ylabel('Normalized Intensity (a.u.)')
        x_ticks=plt.MultipleLocator(5)
        ax=plt.gca()
        ax.set_aspect(1./ax.get_data_ratio())
        ax.xaxis.set_major_locator(x_ticks)
        plt.show()


        # %%
        theta_m=np.zeros(len(totalwavelength))
        phasevariation=np.zeros(len(totalwavelength))



        #resolution
        effective_pixelnum=256
        pixelsize_grating=centralwavelength*FL_fourierlens/(shape_[0]*pixelsize_SLM)   #grating pixel size
        pixelsize_pattern=centralwavelength*objective_FL/(shape[0]*pixelsize_SLM)#pixel size pattern

        #grid
        x=np.linspace((0-shape_[0]//2+1),shape_[0]-shape_[0]//2,shape_[0])# Lateral region of grating (m)
        y=np.linspace((0-shape_[0]//2+1),shape_[0]-shape_[0]//2,shape_[0])
        xx, yy = np.meshgrid(x, y)

        xx=np.float32(xx)
        yy=np.float32(yy)

        #beamsize_y=2.12e-3

        Mask_grating=np.exp(-((xx*pixelsize_grating)**2+(yy*pixelsize_grating)**2)/((beamsize/2))**2)/(beamsize/2)#(xx*pixelsize_grating)**2+(yy*pixelsize_grating)**2<=(beamsize/2)**2
        Mask_grating=Mask_grating.astype(np.float32)
        Mask_grating=(Mask_grating-Mask_grating.min())/(Mask_grating.max()-Mask_grating.min())
        plt.figure()


        plt.pcolor(x*pixelsize_grating*1e3,y*pixelsize_grating*1e3,Mask_grating)
        #plt.axis('equal')
        x_ticks=plt.MultipleLocator(10)
        y_ticks=plt.MultipleLocator(10)
        ax=plt.gca()
        ax.set_aspect(1./ax.get_data_ratio())
        ax.xaxis.set_major_locator(x_ticks)
        ax.yaxis.set_major_locator(y_ticks)
        plt.title('beam on grating')
        plt.xlabel('(mm)')
        plt.ylabel('(mm)')
        plt.xlim(-12.5,12.5)
        plt.ylim(-12.5,12.5)
        #plt.xlabel(u"\u03bcm")
        #plt.ylabel(u"\u03bcm")
        plt.show()

        #output angle from grating each wavelength
        for i in range (len(theta_m)):
            theta_m[i]=np.arcsin(totalwavelength[i]/groove-np.sin(incident_angle))

        result=np.zeros((shape[0],shape[1],len(theta_m)),dtype=np.complex64)
        for i in range (len(theta_m)):
            #grating effect
            phasevariation=np.exp(1j*2*np.pi/totalwavelength[i]*np.sin(theta_m[i])*(xx*pixelsize_grating))
            phasevariation.astype(np.complex64)
            complex_amp=Mask_grating*phasevariation

            #fourier lens to SLM
            complex_amp=np.fft.fftshift(np.fft.fft2(complex_amp))*(wavelength_distribution[i])
            result[:,:,i]=complex_amp[shape_[1]//2-shape[1]//2:shape_[1]//2+shape[1]//2,shape_[1]//2-shape[1]//2:shape_[1]//2+shape[1]//2]

            #intensity[:,:,i]=np.power(np.abs(result[:,:,i]),2)
            #plt.matshow(np.power(np.abs(result[:,:,i]),2))
            #plt.show()
        #intensity=(intensity-intensity.min())/(intensity.max()-intensity.min()) 
        #plt.matshow(intensity.sum(axis=2))
        print("result=",result.dtype)  
        plt.matshow(np.power(np.abs(result),2).sum(axis=2))
        plt.show()
        scio.savemat('input.mat', {'amp':result})
        #del intensity
        del complex_amp
        del phasevariation
        del Mask_grating
        del wavelength_distribution
        del xx
        del yy

        R=np.abs(result)
        del result
        gc.collect()
        #O=np.angle(result)
        '''x=[]
        x.append(R)
        x.append(O)'''


        return R
        
    
    def propagate(phi_slm):
        frames = []
        cf_slm = __phi_slm(phi_slm)
        back_aperture=circle_mask(shape)
        back_aperture=back_aperture.astype(np.float32)
        back_aperture_mask=  tf.broadcast_to(tf.expand_dims(tf.keras.backend.constant(back_aperture, dtype = tf.complex64), axis=0), tf.shape(cf_slm))
        print(type(back_aperture_mask))
        cf_slm*=back_aperture_mask
        mask_amplitude=amplitude_SLM(shape,lambda_,ps)
        mask_amplitude=mask_amplitude.astype(np.complex64)
        for H, z in zip(Hs, zs):
            single_layer=[] 
            for i in range(np.size(mask_amplitude,2)):
                buffer=cf_slm
                TFR=tf.broadcast_to(tf.expand_dims(tf.keras.backend.constant(mask_amplitude[:,:,i], dtype = tf.complex64), axis=0), tf.shape(cf_slm))
                buffer*=TFR
                #print("buffer=",buffer.shape.as_list())
                single_layer.append(__prop__(buffer, tf.keras.backend.constant(H, dtype = tf.complex64)))
            single_layer_multiwavelength=tf.concat(values=single_layer,axis=-1)
            print("single_layer",single_layer_multiwavelength.shape.as_list())
            temporal_focusing=TF_reconstruction(single_layer_multiwavelength)
            frames.append(temporal_focusing)
            del temporal_focusing
            gc.collect()
        y_pred = tf.concat(values=frames, axis = -1)
        print(y_pred)
        y_pred = y_pred/tf.reshape(tf.reduce_max(y_pred,axis=(1,2,3)), (-1, 1, 1, 1))
        #print("y_pred",y_pred.shape.as_list())
        return y_pred
    return propagate

def accuracy(y_true, y_pred):
    denom = tf.sqrt(tf.reduce_sum(tf.pow(y_pred, 2), axis=[1, 2, 3])*tf.reduce_sum(tf.pow(y_true, 2), axis=[1, 2, 3]))
    return tf.reduce_mean((tf.reduce_sum(y_pred * y_true, axis=[1, 2, 3])+1)/(denom+1), axis = 0)

# %%
@tf.function
def __normalize_minmax(img):
    img -= tf.reduce_min(tf.cast(img, tf.float32), axis=[0, 1], keepdims=True)
    img /= tf.reduce_max(img, axis=[0, 1], keepdims=True)
    return img

@tf.function
def __gs(img):
    rand_phi = tf.random.uniform(img.shape)
    img = __normalize_minmax(img)
    img_cf = tf.complex(img, 0.) * tf.math.exp(tf.complex(0., rand_phi))
    slm_cf = tf.signal.ifft2d(tf.signal.ifftshift(img_cf))
    slm_phi = tf.math.angle(slm_cf)
    return slm_phi


def __accuracy(y_true, y_pred):
    denom = tf.sqrt(tf.reduce_sum(tf.pow(y_pred, 2), axis = [0, 1])*tf.reduce_sum(tf.pow(y_true, 2), axis = [0, 1]))
    return 1 - (tf.reduce_sum(y_pred * y_true, axis = [0, 1])+1)/(denom+1)


def novocgh2D(img, Ks, lr = 0.01):
    slms = []
    amps = []
    phi = __gs(img)
    phi_slm = tf.Variable(phi)
    opt = tf.keras.optimizers.Adam(learning_rate = lr)
    img = tf.convert_to_tensor(img)

    def loss_(phi_slm):
        slm_cf = tf.math.exp(tf.complex(0., phi_slm))
        img_cf = tf.signal.fftshift(tf.signal.fft2d(slm_cf))
        return tf.math.abs(img_cf)

    def loss():
        slm_cf = tf.math.exp(tf.complex(0., phi_slm))
        img_cf = tf.signal.fftshift(tf.signal.fft2d(slm_cf))
        amp = tf.math.abs(img_cf)
        return __accuracy(tf.square(img), tf.square(amp))

    for i in range(Ks[-1]+1):
        opt.minimize(loss, var_list=[phi_slm])
        if i in Ks:
            amps.append(loss_(phi_slm).numpy())
            slms.append(phi_slm.numpy())
    return slms, amps

# %%
























