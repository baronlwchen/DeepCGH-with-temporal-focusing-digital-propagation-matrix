#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
"""
The code is modified from "DeepCGH: 3D computer generated holography using deep learning",
including the temporal focusing simulation and digital propagation parts
"""
import os
import scipy.io as sio
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from queue import Queue
from threading import Thread
import warnings
from skimage.draw import circle, line_aa
import numpy as np
from tqdm import tqdm
import h5py as h5
from datetime import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Add, Input, Concatenate, Lambda
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D
import numpy.matlib
import math
import cmath
import matplotlib.pyplot as plt
import gc

# %%
class DeepCGH_Datasets(object):
    '''
    Class for the Dataset object used in DeepCGH algorithm.
    Inputs:
        path   string, determines the lcoation that the datasets are going to be stored in
        num_iter   int, determines the number of iterations of the GS algorithm
        input_shape   tuple of shape (height, width)
    Returns:
        Instance of the object
    '''
    def __init__(self, params):
        try:
            assert params['object_type'] in ['Disk','Gaussian', 'Line', 'Dot'], 'Object type not supported'
            
            self.path = params['path']
            self.shape = params['shape']
            self.N = params['N']
            self.ratio = params['train_ratio']
            self.object_size = params['object_size']
            self.intensity = params['intensity']
            self.object_count = params['object_count']
            self.name = params['name']
            self.object_type = params['object_type']
            self.centralized = params['centralized']
            self.normalize = params['normalize']
            self.compression = params['compression']
            self.layer = params['layer']
            self.wavelength = params['wavelength']
            self.pixel_size = params['pixel_size']
            self.beam_size = params['beam_size']
            self.groove = params['groove']
            self.focal_length_objective = params['focal_length_objective']
            self.focal_length_FL = params['focal_length_FL']
            
            
        except:
            assert False, 'Not all parameters are provided!'
            
        self.__check_avalability()
        
    
    def __check_avalability(self):
        print('Current working directory is:')
        print(os.getcwd(),'\n')
        
        self.filename = self.object_type + '{}_SHP{}_N{}_SZ{}_INT{}_Crowd{}_CNT{}_Split.tfrecords'.format(self.layer, self.shape, 
                                           self.N, 
                                           self.object_size,
                                           self.intensity, 
                                           self.object_count,
                                           self.centralized)
        
        self.absolute_file_path = os.path.join(os.getcwd(), self.path, self.filename)
        if not (os.path.exists(self.absolute_file_path.replace('Split', '')) or os.path.exists(self.absolute_file_path.replace('Split', 'Train')) or os.path.exists(self.absolute_file_path.replace('Split', 'Test'))):
            warnings.warn('File does not exist. New dataset will be generated once getDataset is called.')
            print(self.absolute_file_path)
        else:
            print('Data already exists.')
           
            
    def loadmat():
        # Load the MATLAB .mat file
        mat_file2 = h5.File('Gaussian.mat')
        Gaussian = mat_file2['Gaussian']
        mat_file1 = h5.File('Targets.mat')
        Targets = mat_file1['Targets']
        Gaussian = np.array(Gaussian)
        Gaussian = Gaussian.transpose(3,2,1,0)
        Targets = np.array(Targets)
        Targets = Targets.transpose(3,2,1,0)
        print(Gaussian.shape)
        return Gaussian, Targets
    
    def get_DPM(self, sample):
        sample1=sample
        star=(self.shape[-1]-1)/2*self.layer
        img1=np.zeros((self.shape[0],self.shape[1]))
        for k in range(self.shape[-1]):
            for l in range(self.shape[1]):
                for m in range(self.shape[-1]):
                    if sample1[l,m,k]<0.2:
                        sample1[l,m,k]=0
            img1+=sample1[:,:,k] #疊成第一個矩陣
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[-1]):
                    if sample1[i,j,k]>=0.2:
                        sample1[i,j,k]=-star+k*self.layer #給深度資訊
        img2=np.zeros((self.shape[0],self.shape[1]))
        for k in range(self.shape[-1]):
            img2+=sample1[:,:,k] #將有深度資訊的疊成第二個矩陣
        DPM=np.zeros((sample.shape[0],sample.shape[1],2));  DPM[:,:,0]=img1; DPM[:,:,1]=img2;
        return DPM
    
    
    
    def __generate(self):
        '''
        Creates a dataset of randomly located blobs and stores the data in an TFRecords file. Each sample (3D image) contains
        a randomly determined number of blobs that are randomly located in individual planes.
        Inputs:
            filename : str
                path to the dataset file
            N: int
                determines the number of samples in the dataset
            fraction : float
                determines the fraction of N that is used as "train". The rest will be the "test" data
            shape: (int, int)
                tuple of integers, shape of the 2D planes
            maxnum: int
                determines the max number of blobs
            radius: int
                determines the radius of the blobs
            intensity : float or [float, float]
                intensity of the blobs. If a scalar, it's a binary blob. If a list, first element is the min intensity and
                second one os the max intensity.
            normalize : bool
                flag that determines whether the 3D data is normalized for fixed energy from plane to plane
    
        Outputs:
            aa:
    
            out_dataset:
                numpy.ndarray. Numpy array with shape (samples, x, y)
        '''
        
#        assert self.shape[-1] > 1, 'Wrong dimensions {}. Number of planes cannot be {}'.format(self.shape, self.shape[-1])
        
        train_size = np.floor(self.ratio * self.N)
        # TODO multiple tfrecords files to store data on. E.g. every 1000 samples in one file
        options = tf.io.TFRecordOptions(compression_type = self.compression)
#        options = None
        Gaussian, Targets = DeepCGH_Datasets.loadmat()
        with tf.io.TFRecordWriter(self.absolute_file_path.replace('Split', 'Train'), options = options) as writer_train:
            with tf.io.TFRecordWriter(self.absolute_file_path.replace('Split', 'Test'), options = options) as writer_test:
                for i in tqdm(range(self.N)):
                    sample = Gaussian[:,:,:,i];
                    image_raw = sample.tostring()
                    sample1=sample.copy()
                    DPM = self.get_DPM(sample1);
                    DPM_raw = DPM.tostring()
                    sample1 = Targets[:,:,:,i];
                    Targets_raw = sample1.tostring()
                    #feature = {'sample': self.__bytes_feature(image_raw)}
                    #==============
                    feature = {
                        "Matrix":tf.train.Feature(bytes_list=tf.train.BytesList(value=[DPM_raw])),
                        "Sample":tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))
                    }
                    #==============
                    # 2. Create a tf.train.Features
                    features = tf.train.Features(feature = feature)
                    # 3. Create an example protocol
                    example = tf.train.Example(features = features)
                    # 4. Serialize the Example to string
                    example_to_string = example.SerializeToString()
                    # 5. Write to TFRecord
                    if i < train_size:
                        writer_train.write(example_to_string)
                    else:
                        writer_test.write(example_to_string)
            
    
    def getDataset(self):
        if not (os.path.exists(self.absolute_file_path.replace('Split', '')) or os.path.exists(self.absolute_file_path.replace('Split', 'Train')) or os.path.exists(self.absolute_file_path.replace('Split', 'Test'))):
            print('Generating data...')
            folder = os.path.join(os.getcwd(), self.path)
            
            if not os.path.exists(folder):
                os.makedirs(folder)
            
            self.__generate()
        self.dataset_paths = [self.absolute_file_path.replace('Split', 'Train'), self.absolute_file_path.replace('Split', 'Test')]
        
        
#=======================================================================================================================================================
        

        
        

        

# %%
class DeepCGH(object):
    '''
    Class for the DeepCGH algorithm.
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
        
        self.filename = model_params['filename']
        self.path = model_params['path']
        self.shape = data_params['shape']
        self.object_size = data_params['object_size']
        self.plane_distance = model_params['plane_distance']
        self.n_kernels = model_params['n_kernels']
        self.IF = model_params['int_factor']
        self.wavelength = model_params['wavelength']
        self.f = model_params['focal_length']
        self.ps = model_params['pixel_size']
        self.object_type = data_params['object_type']
        self.centralized = data_params['centralized']
        self.input_name = model_params['input_name']
        self.output_name = model_params['output_name']
        self.token = model_params['token']
        self.zs = np.linspace(0-(self.shape[-1]//2),self.shape[-1]//2,self.shape[-1])*self.plane_distance
        #[-1*self.plane_distance*x for x in np.arange(1, (self.shape[-1]-1)//2+1)][::-1] + [self.plane_distance*x for x in np.arange(1, (self.shape[-1]-1)//2+1)]
        self.input_queue = Queue(maxsize=4)
        self.output_queue = Queue(maxsize=4)
        self.__check_avalability()
        self.lr = model_params['lr']
        self.batch_size = model_params['batch_size']
        self.epochs = model_params['epochs']
        self.token = model_params['token']
        self.shuffle = model_params['shuffle']
        self.max_steps = model_params['max_steps']
        self.n=model_params['refraction_index']
        
        
    def __start_thread(self):
        self.prediction_thread = Thread(target=self.__predict_from_queue, daemon=True)
        self.prediction_thread.start()
        
        
    def __check_avalability(self):
        print('Looking for trained models in:')
        print(os.getcwd(), '\n')
        
        self.filename = '{}Model_{}_SHP{}_IF{}_WL{}_PS{}_CNT{}_{}'.format(self.filename,
                                                                              self.object_type,
                                                                              (self.shape[0], self.shape[1], 2), 
                                                                              self.IF,
                                                                              self.wavelength,
                                                                              self.ps,
                                                                              self.centralized,
                                                                              self.token)
        
        self.absolute_file_path = os.path.join(os.getcwd(), self.path, self.filename)
        
        if not os.path.exists(self.absolute_file_path):
            print('No trained models found. Please call the `train` method. \nModel checkpoints will be stored in: \n {}'.format(self.absolute_file_path))
            
        else:
            print('Model already exists.')
    
    
    def __make_folders(self):
        if not os.path.exists(self.absolute_file_path):
            os.makedirs(self.absolute_file_path)
    
        
    def train(self, deepcgh_dataset, lr = None, batch_size = None, epochs = None, token = None, shuffle = None, max_steps = None):
        # Using default params or new ones?
        if lr is None:
            lr = self.lr
        if batch_size is None:
            batch_size = self.batch_size
        if epochs is None:
            epochs = self.epochs
        if token is None:
            token = self.token
        if shuffle is None:
            shuffle = self.shuffle
        if max_steps is None:
            max_steps = self.max_steps
        
        # deifne Estimator
        model_fn = self.__get_model_fn()
        
        # Data
        train, validation = self.load_data(deepcgh_dataset.dataset_paths, batch_size, epochs, shuffle)
        
        self.__make_folders()
        self.config = tf.estimator.RunConfig(log_step_count_steps=50,save_checkpoints_steps=400)
        self.estimator = tf.estimator.Estimator(model_fn,
                                                model_dir=self.absolute_file_path,
                                               config=self.config)
        
        # 
        train_spec = tf.estimator.TrainSpec(input_fn=train, max_steps=max_steps)
        eval_spec = tf.estimator.EvalSpec(input_fn=validation)
        
        tf.estimator.train_and_evaluate(self.estimator, train_spec, eval_spec)

        self.__start_thread()
            
    
    def load_data(self, path, batch_size, epochs, shuffle):
        if isinstance(path, list) and ('tfrecords' in path[0]) and ('tfrecords' in path[1]):
            image_feature_description = {'Sample': tf.io.FixedLenFeature([], tf.string),'Matrix': tf.io.FixedLenFeature([], tf.string)}
            
            #DPM_description = {}
            def __parse_image_function(example_proto):
                parsed_features = tf.io.parse_single_example(example_proto, image_feature_description)
                img = tf.cast(tf.reshape(tf.io.decode_raw(parsed_features['Sample'], tf.float64), self.shape), tf.float32)
                img1 = tf.cast(tf.reshape(tf.io.decode_raw(parsed_features['Matrix'], tf.float64), (self.shape[0], self.shape[1], 2)), tf.float32)
                return {'target':img1}, {'recon':img}
            '''---------------------------------------------------------------------
            def __parse_image_function(example_proto):
                parsed_features = tf.io.parse_single_example(example_proto, image_feature_description)
                img = tf.cast(tf.reshape(tf.io.decode_raw(parsed_features['Sample'], tf.float64), self.shape), tf.float32)
                img1 = tf.cast(tf.reshape(tf.io.decode_raw(parsed_features['Matrix'], tf.float64), (self.shape[0], self.shape[1], 2)), tf.float32)
                img2 = tf.cast(tf.reshape(tf.io.decode_raw(parsed_features['Target'], tf.uint8), self.shape), tf.float32)
                return {'target':img1, 'new_feature':img2}, {'recon':img}
            
            '''
            
            
            
            def val_func():
                validation = tf.data.TFRecordDataset(path[1],
                                      compression_type='GZIP',
                                      buffer_size=None,
                                      num_parallel_reads=2).map(__parse_image_function).batch(batch_size)#.prefetch(tf.data.experimental.AUTOTUNE)
                
                return validation
            
            def train_func():
                train = tf.data.TFRecordDataset(path[0],
                                      compression_type='GZIP',
                                      buffer_size=None,
                                      num_parallel_reads=2).map(__parse_image_function).repeat(epochs).shuffle(shuffle).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
                return train
            #.repeat(epochs)
            return train_func, val_func
        else:
            raise('You got a problem in your file name bruh')
    
        
    def __generate_from_queue(self):
        '''
        A generator with infinite loop to fetch one smaple from the queue
        Returns:
            one sample!
        '''
        while True:
            yield self.input_queue.get()


    def __predict_from_queue(self):
        '''
        Once the input queue has something to offer to the estimator, the
        estimator makes the predictions and the outputs of that prediction are
        fed into the output queue.
        '''
        for i in self.estimator.predict(input_fn=self.__queued_predict_input_fn,
                                        yield_single_examples=False):
            self.output_queue.put(i)
        
    
    def get_hologram(self, inputs):
        '''
        Return the hologram using the GS algorithm with num_iter iterations.
        Inputs:
            inputs   numpy ndarray, the two dimentional target image
        Returns:
            hologram as a numpy ndarray 
        '''
        features = {}
        #print("x")
        if not isinstance(self.input_name, str):
            for key, val in zip(self.input_name, inputs):
                features[key] = val
        else:
            features = {self.input_name: inputs}
        self.input_queue.put(features)
        predictions = self.output_queue.get()
        
        return predictions#[self.output_name]

    def __queued_predict_input_fn(self):
        '''
        Input function that returns a tensorflow Dataset from a generator.
        Returns:
            a tensorflow dataset
        '''
        # Fetch the inputs from the input queue
        type_dict = {}
        shape_dict = {}
        
        if not isinstance(self.input_name, str):
            for key in self.input_name:
                type_dict[key] = tf.float32
                shape_dict[key] = (None,)+(self.shape[0], self.shape[1],2)
        else:
            type_dict = {self.input_name: tf.float32}
            shape_dict = {self.input_name:(None,)+(self.shape[0], self.shape[1],2)}
        
        dataset = tf.data.Dataset.from_generator(self.__generate_from_queue,
                                                 output_types=type_dict,
                                                 output_shapes=shape_dict)
        return dataset
    
    
    def __get_model_fn(self):
        
        def interleave(x):
            return tf.nn.space_to_depth(input = x,
                                       block_size = self.IF,
                                       data_format = 'NHWC')
        
        
        def deinterleave(x):
            return tf.nn.depth_to_space(input = x,
                                       block_size = self.IF,
                                       data_format = 'NHWC')
        
        
        def __cbn(ten, n_kernels, act_func):
            x1 = Conv2D(n_kernels, (3, 3), activation = act_func, padding='same')(ten)
            x1 = BatchNormalization()(x1)
            x1 = Conv2D(n_kernels, (3, 3), activation = act_func, padding='same')(x1)
            x1 = BatchNormalization()(x1)
            return x1 
        
        
        def __cc(ten, n_kernels, act_func):
            x1 = Conv2D(n_kernels, (3, 3), activation = act_func, padding='same')(ten)
            x1 = Conv2D(n_kernels, (3, 3), activation = act_func, padding='same')(x1)
            return x1
        
        
        def __get_H(zs, shape, lambda_, ps,f,n):
            Hs = []
            #radius=shape[0]//4
            #location=(shape[0]//2+1,shape[1]//2+1)
            k=np.pi*2/lambda_*n
            #print(zs)
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
        
        def circle_mask(shape):
            radius=485
            #for 20X/0.25 objective
            location=(shape[0]//2+1,shape[1]//2+1)
            img = np.zeros(shape[:-1], dtype=np.float32)
            rr, cc = circle(location[0], location[1], radius, shape=img.shape)
            img[rr, cc] = 1
            #plt.plot(rr,cc)
            return img
        
        def __unet():
            n_kernels = self.n_kernels
            shape = self.shape
            inp = Input(shape=(self.shape[0], self.shape[1],2), name='target')
            act_func = 'relu'
            x1_1 = Lambda(interleave, name='Interleave')(inp)
            # Block 1
            x1 = __cbn(x1_1, n_kernels[0], act_func)
            x2 = MaxPooling2D((2, 2), padding='same')(x1)
            # Block 2
            x2 = __cbn(x2, n_kernels[1], act_func)
            encoded = MaxPooling2D((2, 2), padding='same')(x2)
            # Bottleneck
            encoded = __cc(encoded, n_kernels[2], act_func)
            #
            x3 = UpSampling2D(2)(encoded)
            x3 = Concatenate()([x3, x2])
            x3 = __cc(x3, n_kernels[1], act_func)
            #
            x4 = UpSampling2D(2)(x3)
            x4 = Concatenate()([x4, x1])
            x4 = __cc(x4, n_kernels[0], act_func)
            #
            x4 = __cc(x4, n_kernels[1], act_func)
            x4 = Concatenate()([x4, x1_1])
            #
            phi_0_ = Conv2D(self.IF**2, (3, 3), activation=None, padding='same')(x4)
            phi_0 = Lambda(deinterleave, name='phi_0')(phi_0_)
            amp_0_ = Conv2D(self.IF**2, (3, 3), activation='relu', padding='same')(x4)
            amp_0 = Lambda(deinterleave, name='amp_0')(amp_0_)
            #Gauss=apply_Gaussian_on_phase(self.shape)
            #amp_0*=Gauss
            #Gauss = tf.broadcast_to(tf.expand_dims(Gauss, axis=0), tf.shape(cf_slm))
            phi_slm = Lambda(__ifft_AmPh, name='phi_slm')([amp_0, phi_0, shape])
            
            return Model(inp, phi_slm)
            
        
        
        
        def __ifft_AmPh(x):
            '''
            Input is Amp x[1] and Phase x[0]. Spits out the angle of ifft.
            '''
            shapes=x[2]
            img = tf.dtypes.complex(tf.squeeze(x[0], axis=-1), 0.) * tf.math.exp(tf.dtypes.complex(0., tf.squeeze(x[1], axis=-1)))
            img = tf.signal.ifftshift(img, axes = [1, 2])
            fft = tf.multiply(tf.signal.ifft2d(img),tf.cast(shapes[1],dtype=tf.dtypes.complex64))
            phase = tf.expand_dims(tf.math.angle(fft), axis=-1)
            
            return phase
        
        def amplitude_SLM(shape,centralwavelength,pixelsize_SLM):
            M=0.75
            focal_length_fourierlens=750e-3#*M
            #(focal_length)
            beamsize=4.24e-3*150/250
            C=3e8
            centralfrequency=C/centralwavelength
            pulse_duration=228e-15
            groove=1/600*1e-3
            N_slits=beamsize/groove
            #wavelength_increment=centralwavelength/(1*N_slits)
            shape_=(shape[0]*8,shape[1]*8)
            print(shape_)
            bandwidth=0.441/pulse_duration/C*np.power(centralwavelength,2)
            sampling_frequency=1.6e11
            spectral_bandwidth=C/np.power(centralwavelength,2)*bandwidth
            incident_angle=np.arcsin(centralwavelength/groove)
            totalfrequency=np.arange(centralfrequency-spectral_bandwidth,centralfrequency+spectral_bandwidth,sampling_frequency)
            theta_m=np.zeros(len(totalfrequency))
            phasevariation=np.zeros(len(totalfrequency))
            for i in range (len(theta_m)):
                theta_m[i]=np.arcsin((C/totalfrequency[i])/groove-np.sin(incident_angle))



            pixelsize_grating=centralwavelength*focal_length_fourierlens/(shape_[0]*pixelsize_SLM)
           
            x=np.linspace((0-shape_[0]//2+1),shape_[0]-shape_[0]//2,shape_[0]);# Lateral region of grating (m)
            
            xx, yy = np.meshgrid(x, x)
            xx=np.float32(xx)
            yy=np.float32(yy)
            
            Mask_grating_intensity=np.exp(-2*(((xx*pixelsize_grating)**2+(yy*pixelsize_grating)**2)/(beamsize/2)**2))
            #Mask_grating_intensity=(Mask_grating_intensity-Mask_grating_intensity.min())/(Mask_grating_intensity.max()-Mask_grating_intensity.min())
            Mask_grating=np.sqrt(Mask_grating_intensity)
            Mask_grating=Mask_grating.astype(np.float32)
            #plt.matshow(Mask_grating)
            #plt.show()

            mean=np.mean(totalfrequency)
            sigma=spectral_bandwidth/(2*np.sqrt(2*np.log(2)))
            frequency_distribution=np.sqrt(np.pi/(2*np.log(2)))*pulse_duration*np.exp(-1*np.power(pulse_duration,2)*np.power((totalfrequency-mean)*2*np.pi,2)/(8*np.log(2)))
            frequency_distribution=frequency_distribution/max(frequency_distribution)
            plt.plot(totalfrequency,frequency_distribution)

            result=np.zeros((shape[0],shape[1],len(theta_m)),dtype=np.complex64)
            #intensity=np.zeros((shape[0],shape[1],len(theta_m)))
            for i in range (len(theta_m)):
                phasevariation=np.exp(1j*2*np.pi/(C/totalfrequency[i])*np.sin(theta_m[i])*(xx*pixelsize_grating))
                phasevariation.astype(np.complex64)
                complex_amp=Mask_grating*phasevariation
                complex_amp=np.fft.fftshift(np.fft.fft2(complex_amp)/shape_[1])
                result[:,:,i]=complex_amp[shape_[1]//2-shape[1]//2:shape_[1]//2+shape[1]//2,shape_[1]//2-shape[1]//2:shape_[1]//2+shape[1]//2]*frequency_distribution[i]
                
                #intensity[:,:,i]=np.power(np.abs(result[:,:,i]),2)
                #plt.matshow(np.power(np.abs(result[:,:,i]),2))
                #plt.show()
            #intensity=(intensity-intensity.min())/(intensity.max()-intensity.min()) 
            #plt.matshow(intensity.sum(axis=2))
            print("result=",result.dtype)  
            plt.matshow(np.power(np.abs(result),2).sum(axis=2))
            plt.show()
            #del intensity
            del complex_amp
            del phasevariation
            del Mask_grating
            del frequency_distribution
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
        
        def rescale(X, a, b):
            repeat = X.shape[1]
            xmin = tf.repeat(tf.reshape(tf.math.reduce_min(X, axis=1), shape=[-1,1]), repeats=repeat, axis=1)
            xmax = tf.repeat(tf.reshape(tf.math.reduce_max(X, axis=1), shape=[-1,1]), repeats=repeat, axis=1)
            X = (X - xmin) / (xmax-xmin)
            return X * (b - a) + a
        
        #zero-padding
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
                result_=tf.reduce_sum(tf.pow(tf.abs(tf.signal.fftshift(tf.divide(tf.signal.fft(result),32),axes=(3))),4),axis=-1,keepdims=True)
                #print("timedomain=",result_.shape.as_list())
                single_layer.append(result_)
            imgTF=tf.cast(tf.concat(values=single_layer,axis=1),dtype=tf.dtypes.float32)
            #imgTF_ = imgTF  / tf.reshape(tf.reduce_max(imgTF,axis=(1,2,3)), (-1, 1, 1, 1))
            #imgTF_ = (imgTF - tf.reshape(tf.reduce_min(imgTF, axis=(1, 2, 3)), (-1, 1, 1, 1))) / (tf.reshape(tf.reduce_max(imgTF,axis=(1,2,3)), (-1, 1, 1, 1)) - tf.reshape(tf.reduce_min(imgTF, axis=(1, 2, 3)), (-1, 1, 1, 1)))
            #print("imgTF=",imgTF.shape.as_list())
            
            #imgTF_ = (imgTF - tf.reshape(tf.reduce_min(imgTF, axis=(1, 2, 3)), (-1, 1, 1, 1))) / (tf.reshape(tf.reduce_max(imgTF,axis=(1,2,3)), (-1, 1, 1, 1)) - tf.reshape(tf.reduce_min(imgTF, axis=(1, 2, 3)), (-1, 1, 1, 1)))
            #del imgTF
            del result
            del single_layer
            del temporal_focusing
            del frequency_component
            gc.collect()
               
            return imgTF
        
        def phase_unwrapping(single_layer_multiwavelength_angle):
            threshold=math.pi
            #condition=tf.math.greater_equal(single_layer_multiwavelength_angle[:,:,k+1]-single_layer_multiwavelength_angle[:,:,k],math.pi)
            for k in range(single_layer_multiwavelength_angle.shape[3]-1):
                #print(k)
                condition1=tf.math.greater_equal(single_layer_multiwavelength_angle[:,:,:,k+1]-single_layer_multiwavelength_angle[:,:,:,k],math.pi)

                count=0
                buffer=tf.cast(single_layer_multiwavelength_angle[:,:,:,k+1],dtype=tf.float32)
                comparison=single_layer_multiwavelength_angle[:,:,:,k]
                '''
                a=single_layer_multiwavelength_angle[:,:,k+1:]
                print(a.shape.as_list())
                '''
                while tf.reduce_any(condition1):
                    update=tf.where(condition1,buffer-2*math.pi,buffer)
                    #update=tf.cast(buffer,dtype=tf.float32)
                    #print(count)
                    #count=count+1
                #print(single_layer_multiwavelength_angle[:,:,0:k+1].shape.as_list())
                    condition2=tf.math.greater_equal(comparison-update,math.pi)
                    while tf.reduce_any(condition2):
                        update=tf.where(condition2,update+2*math.pi,update)
                        condition2=tf.math.greater(comparison-update,math.pi)
                    condition1=tf.math.greater(update-comparison,math.pi)
                    buffer=tf.cast(update,dtype=tf.float32)

                    #print("single_layer_multiwavelength_angle[:,:,0:k]",single_layer_multiwavelength_angle[:,:,k].shape.as_list())

                single_layer_multiwavelength_angle=tf.concat([single_layer_multiwavelength_angle[:,:,:,:k+1],tf.expand_dims(update,axis=-1),single_layer_multiwavelength_angle[:,:,:,k+2:]],axis=-1)
                #single_layer_multiwavelength_angle_unwrapped.append(update)
                #single_layer_multiwavelength_angle_unwrapped_=tf.concat(values=single_layer_multiwavelength_angle_unwrapped,axis=-1)
                #print(single_layer_multiwavelength_angle.shape.as_list())
            '''

            for i in range(1):
                for j in range(1):
                    buffer=[]
                    buffer.append(single_layer_multiwavelength_angle[i,j,0])
                    for k in range(single_layer_multiwavelength_angle.shape[2]-1):
                        if single_layer_multiwavelength_angle[i,j,k+1]-single_layer_multiwavelength_angle[i,j,k]>threshold:
                            buffer.append(single_layer_multiwavelength_angle[i,j,k+1:single_layer_multiwavelength_angle.shape[2]-1]-2*math.pi)
                        elif single_layer_multiwavelength_angle[i,j,k+1]-single_layer_multiwavelength_angle[i,j,k]<threshold:
                            buffer.append(single_layer_multiwavelength_angle[i,j,k+1:single_layer_multiwavelength_angle.shape[2]-1]+2*math.pi)
                        else:
                            buffer.append(single_layer_multiwavelength_angle[i,j,k+1:single_layer_multiwavelength_angle.shape[2]-1])
                    a=tf.concat(values=buffer,axis=0)
                    print(a.shape.as_list())
            '''
            return single_layer_multiwavelength_angle
            
        
        
        
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
            
            
            
        def __prop__(cf_slm, H):
            H = tf.broadcast_to(tf.expand_dims(H, axis=0), tf.shape(cf_slm))
            cf_slm *=H
            complex_amp_focal_plane=tf.signal.fftshift(tf.divide(tf.signal.fft2d(cf_slm),cf_slm.shape[1]),axes=[1,2])
            #img=tf.expand_dims(complex_amp_focal_plane, axis=-1)
            img=tf.cast(tf.expand_dims(complex_amp_focal_plane, axis=-1), dtype=tf.dtypes.complex64)
            #print("amp_onfocal",img.shape.as_list())
            
            #fft=tf.signal.fftshift(tf.signal.fft2d(cf_slm),axes=[1,2])
            #A=tf.signal.fftshift(tf.signal.fft2d(fft),axes=[1,2])
            #if not center:
            #    H = tf.broadcast_to(tf.expand_dims(H, axis=0), tf.shape(cf_slm))
            #    #cf_slm *= tf.signal.fftshift(H, axes = [1, 2])
            #    A*=H
            #U=tf.signal.ifft2d(tf.signal.fftshift(A,axes=[1,2]))
            #fft = tf.signal.ifftshift(tf.signal.fft2d(tf.signal.fftshift(cf_slm, axes = [1, 2])), axes = [1, 2])
            
            #img=tf.expand_dims(U, axis=-1)
            #img=tf.cast(tf.expand_dims(fft, axis=-1), dtype=tf.dtypes.complex64)
            del complex_amp_focal_plane
            gc.collect()
            return img
        
        
        def __phi_slm(phi_slm):
            #print("phi_slm",phi_slm.shape.as_list())
            #i_phi_slm_angle = tf.dtypes.complex(np.float32(0.), tf.squeeze(phi_slm, axis=-1)+TFO)
            #print("i_phi_slm",i_phi_slm.shape.as_list())
            i_phi_slm = tf.dtypes.complex(np.float32(0.), tf.squeeze(phi_slm, axis=-1))
            return tf.math.exp(i_phi_slm)
        
        
        Hs = __get_H(self.zs, self.shape, self.wavelength, self.ps,self.f,self.n)
        input_amp=amplitude_SLM(self.shape,self.wavelength,self.ps)
        #N=5
        #hyperparameter=10113/np.power(np.size(self.zs)*N,2)
        
        def stimulation_acc(y_true_,single_layer_multiwavelength):
            denom = tf.sqrt(tf.reduce_sum(tf.pow(single_layer_multiwavelength, 2), axis=[1, 2])*tf.reduce_sum(tf.pow(y_true_, 2), axis=[1, 2]))
            #print("denom",denom.shape.as_list())
            loss=1-tf.reduce_mean(tf.reduce_mean((tf.reduce_sum(single_layer_multiwavelength * y_true_, axis=[1, 2])+1)/(denom+1),axis=-1), axis = 0)
            return loss
        def __accuracy(y_true, y_pred):
            y = list(range(1,(self.shape[-1]+1)))
            ny1 = [y[-1]] + y[:-1]
            ny2 = y[1:] + [y[0]]
            
            denom = tf.sqrt(tf.reduce_sum(tf.pow(y_pred, 2), axis=[1, 2, 3])*tf.reduce_sum(tf.pow(y_true, 2), axis=[1, 2, 3]))
            loss = (1-tf.reduce_mean((tf.reduce_sum(y_pred * y_true, axis=[1, 2, 3]))/(denom), axis = 0))
            
            y_true1 = tf.gather(y_true, ny1, axis=-1)
            denom1 = tf.sqrt(tf.reduce_sum(tf.pow(y_pred, 2), axis=[1, 2, 3])*tf.reduce_sum(tf.pow(y_true1, 2), axis=[1, 2, 3]))
            loss1 = (tf.reduce_mean((tf.reduce_sum(y_pred * y_true1, axis=[1, 2, 3]))/(denom), axis = 0))
            
            y_true2 = tf.gather(y_true, ny2, axis=-1)
            denom2 = tf.sqrt(tf.reduce_sum(tf.pow(y_pred, 2), axis=[1, 2, 3])*tf.reduce_sum(tf.pow(y_true2, 2), axis=[1, 2, 3]))
            loss2 = (tf.reduce_mean((tf.reduce_sum(y_pred * y_true2, axis=[1, 2, 3]))/(denom), axis = 0))
            
            s_y_pred = tf.pow(y_pred, 0.25)
            loss3 = tf.reduce_mean(tf.reduce_sum(s_y_pred * y_true, axis=[1, 2, 3]))/1000
            '''
            s_y_pred = tf.pow(y_pred, 0.25)
            max1 = tf.reduce_max(s_y_pred, axis=[1, 2, 3])
            min1 = tf.reduce_min(s_y_pred, axis=[1, 2, 3])
            loss3 = tf.reduce_mean((max1-min1)/(max1+min1), axis=0)
            return loss+(1-loss3)
        
            '''
            return (1-loss3)
        
        def __big_loss(y_true, phi_slm):
            frames = []
            cf_slm = __phi_slm(phi_slm)
            back_aperture=circle_mask(self.shape)
            back_aperture=back_aperture.astype(np.float32)
            back_aperture_mask=tf.broadcast_to(tf.expand_dims(tf.keras.backend.constant(back_aperture, dtype = tf.complex64), axis=0), tf.shape(cf_slm))
            cf_slm*=back_aperture_mask
            
            mask_amplitude=input_amp
            mask_amplitude=mask_amplitude.astype(np.complex64)
            
            count=0
            acc_=[]
            target_mask=tf.where(y_true>=0.1, 1, 0)
            total_pixel=tf.reduce_sum(target_mask)
            for H, z in zip(Hs, self.zs):
                #print(z)
                single_layer=[] 
                for i in range(np.size(mask_amplitude,2)):
                    buffer=cf_slm
                    TFR=tf.broadcast_to(tf.expand_dims(tf.keras.backend.constant(mask_amplitude[:,:,i], dtype = tf.complex64), axis=0), tf.shape(cf_slm))
                    buffer*=TFR
                    single_layer.append(__prop__(buffer, tf.keras.backend.constant(H, dtype = tf.complex64)))
                single_layer_multiwavelength=tf.concat(values=single_layer,axis=-1)
                y_true_=tf.broadcast_to(tf.expand_dims(y_true[:,:,:,count],axis=-1), tf.shape(single_layer_multiwavelength))
                #print("y_true_",y_true_.shape.as_list())
                single_layer_multiwavelength_=tf.pow(tf.abs(single_layer_multiwavelength),2)
                acc_.append(stimulation_acc(y_true_,single_layer_multiwavelength_))
                temporal_focusing=TF_reconstruction(single_layer_multiwavelength)
                frames.append(temporal_focusing)
                del temporal_focusing
                gc.collect()
                count=count+1
            y_pred = tf.concat(values=frames, axis = -1)
            
            
            
            acc__=tf.concat(values=tf.expand_dims(acc_,axis=-1),axis=-1)
            print("acc__",acc__.shape.as_list())
            acc_=tf.reduce_mean(acc__,axis=[0,1])
            
            loss3 = 1-tf.reduce_mean(tf.reduce_sum(tf.pow(y_pred, 0.25) * y_true, axis=[1, 2, 3])/1e4)
            
            del frames
            del mask_amplitude
            gc.collect()
            #tf.reduce_mean(tf.reduce_mean(tf.square(tf.math.square(y_true) -  tf.math.divide(y_pred,hyperparameter)),axis=(1,2,3)),axis=0)
            return loss3
            
            
        
        def model_fn(features, labels, mode):
            unet = __unet()
            
            
            training = (mode == tf.estimator.ModeKeys.TRAIN)
            
            phi_slm = unet(features['target'], training = training)
        
            if mode == tf.estimator.ModeKeys.PREDICT:
                return tf.estimator.EstimatorSpec(mode, predictions = phi_slm)
            
            else:
                acc = __big_loss(labels['recon'], phi_slm)
                #acc = __big_loss(labels['recon'], phi_slm,features['new_feature'])
                
                if mode == tf.estimator.ModeKeys.EVAL:
                    return tf.estimator.EstimatorSpec(mode, loss = acc)
                
                elif mode == tf.estimator.ModeKeys.TRAIN:
                    train_op = None
                    
                    opt = tf.keras.optimizers.Adam(learning_rate=self.lr)
                    
                    opt.iterations = tf.compat.v1.train.get_or_create_global_step()
                    
                    update_ops = unet.get_updates_for(None) + unet.get_updates_for(features['target'])
                    
                    minimize_op = opt.get_updates(acc , unet.trainable_variables)[0]
                    
                    train_op = tf.group(minimize_op, *update_ops)
                    
                    return tf.estimator.EstimatorSpec(mode = mode,
                                                      predictions = {self.output_name: phi_slm},
                                                      loss = acc,
                                                      train_op = train_op)
        return model_fn


