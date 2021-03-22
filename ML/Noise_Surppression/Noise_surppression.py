from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer, flatten, fully_connected
import numpy as np
import keras
from keras.layers import Input, Dense, Conv1D, Conv2D, Conv2DTranspose, BatchNormalization,subtract, Subtract
from keras.layers import LeakyReLU, PReLU, Reshape, Concatenate, Flatten
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
keras_backend = tf.keras.backend
keras_initializers = tf.keras.initializers
from data_ops import *
from file_ops import *
from models import *
from functools import partial
from keras.layers.merge import _Merge
from keras import backend as K
import time
from tqdm import *
import h5py
import os,sys
import scipy.io.wavfile as wavfile
import random
from wgan_ops import *

class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this
    outputs a random point on the line between each pair of input points.
    Inheriting from _Merge is a little messy but it was the quickest solution I could
    think of. Improvements appreciated."""

    def _merge_function(self, inputs):
        weights = K.random_uniform((55, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])

if __name__ == '__main__':


    os.environ["CUDA_VISIBLE_DEVICES"]= '2'
    # Various GAN options
    opts = {}
    opts ['dirhead'] = "Xavier_OLD_G_GAN"
    opts ['z_off'] = True # set to True to omit the latent noise input
    # normalization
    #################################
    # Only one of the follwoing should be set to True
    opts ['applyinstancenorm'] = True
    opts ['applybn'] = False
    ##################################
    # Show model summary
    opts ['show_summary'] = False
    
    ## Set the matfiles
    clean_train_matfile = "./data2/sm/clean_train_segan1d.mat"
    noisy_train_matfile = "./data2/sm/noisy_train_segan1d.mat"
    noisy_test_matfile = "./data2/sm/noisy_test_segan1d.mat"
    
    ####################################################
    # Other fixed options
    opts ['window_length'] =  2**14
    opts ['featdim'] = 1 # 1 since it is just 1d time samples
    opts ['filterlength'] =  31
    opts ['strides'] = 2
    opts ['padding'] = 'SAME'
    opts ['g_enc_numkernels'] = [16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]
    opts ['d_fmaps'] = opts ['g_enc_numkernels'] # We use the same structure for discriminator
    opts['leakyrelualpha'] = 0.3
    opts ['batch_size'] = 50
    opts ['applyprelu'] = True
    opts ['preemph'] = 0.95
    opts ['D_real_target'] = 1. # Use 0.9 0r 0.95 if you want to apply label smoothing
    
    opts ['d_activation'] = 'leakyrelu'
    g_enc_numkernels = opts ['g_enc_numkernels']
    opts ['g_dec_numkernels'] = g_enc_numkernels[:-1][::-1] + [1]
    opts ['gt_stride'] = 2
    opts ['g_l1loss'] = 200.
    opts ['d_lr'] = 0.0002
    opts ['g_lr'] = 0.0002
    opts ['random_seed'] = 111
    min_ = 99999
    
    GRADIENT_PENALTY_WEIGHT = 10
    n_epochs = 81
    fs = 16000
    
    # set flags for training or testing
    TRAIN_MODEL =  True
    SAVE_MODEL =  True
    LOAD_SAVED_MODEL = False
    TEST_MODEL =  True

    modeldir = get_modeldirname(opts)
    print ("The model directory is " + modeldir)
    print ("_____________________________________")

    if not os.path.exists(modeldir):
        os.makedirs(modeldir)

    # Each module
    speech_D = speech_discriminator(opts)
    noise_D = noise_discriminator(opts)
    #T_G = generator(opts)
    S_G = generator(opts)
    
    g_opt = keras.optimizers.Adam(lr=opts['g_lr'])
    d_opt = keras.optimizers.Adam(lr=opts['d_lr']) # Use same value in speech_D and noise_D
    
    wav_shape = (opts['window_length'], opts['featdim'])
    z_dim1 = int(opts['window_length']/ (opts ['strides'] ** len(opts ['g_enc_numkernels'])))
    z_dim2 = opts ['g_enc_numkernels'][-1]
    wav_in_clean =  Input(shape=wav_shape)
    wav_in_noisy = Input(shape=wav_shape)
    
    if not opts ['z_off']:
        z = Input (shape=(z_dim1, z_dim2))
        #T_G_wav = T_G([wav_in_noisy, z])
        S_G_wav = S_G([wav_in_noisy, z])
        #T_G_noise_wav = subtract([wav_in_noisy, T_G_wav])
        S_G_noise_wav = subtract([wav_in_noisy, S_G_wav])
        #T_G = Model([wav_in_noisy, z], T_G_wav)
        S_G = Model([wav_in_noisy, z], S_G_wav)
    else :
        #T_G_wav = T_G(wav_in_noisy)
        #T_G_noise_wav = subtract([wav_in_noisy, T_G_wav])
        #T_G = Model(wav_in_noisy, T_G_wav)
        S_G_wav = S_G(wav_in_noisy)
        S_G_noise_wav = subtract([wav_in_noisy, S_G_wav])
        S_G = Model(wav_in_noisy, S_G_wav)
        
    speech_d_out = speech_D([wav_in_clean, wav_in_noisy])
    speech_d_out = Activation('sigmoid')(speech_d_out)
    
    noise_d_out = noise_D([wav_in_clean, wav_in_noisy])
    noise_d_out = Activation('sigmoid')(noise_d_out)
    
    speech_D = Model([wav_in_clean, wav_in_noisy], speech_d_out)
    noise_D = Model([wav_in_clean, wav_in_noisy], noise_d_out)
    
    
    #T_G.compile(optimizer=g_opt,loss= 'mean_absolute_error')
    S_G.compile(optimizer=g_opt,loss= 'mean_absolute_error')
    speech_D.compile(loss='mean_squared_error', optimizer=d_opt)
    noise_D.compile(loss='mean_squared_error', optimizer=d_opt)
    
    S_G.summary()
    #T_G.summary()
    speech_D.summary()
    noise_D.summary()
    
    #GAN module
    for layer in speech_D.layers :
        layer.trainable = False
    speech_D.trainable = False
    for layer in noise_D.layers :
        layer.trainable = False
    noise_D.trainable = False
   
    #speech_D_out_T = speech_D([T_G_wav, wav_in_noisy])
    #noise_D_out_T = noise_D([T_G_noise_wav, wav_in_noisy])
    speech_D_out_S = speech_D([S_G_wav, wav_in_noisy])
    noise_D_out_S = noise_D([S_G_noise_wav, wav_in_noisy])
    
    if not opts ['z_off']:
        #G_T_D = Model(inputs=[wav_in_clean, wav_in_noisy, z], outputs=[speech_D_out_T, T_G_wav, noise_D_out_T, T_G_noise_wav])
        G_S_D = Model(inputs=[wav_in_clean, wav_in_noisy, z], outputs=[speech_D_out_S, S_G_wav, noise_D_out_S, S_G_noise_wav])
    else :
        #G_T_D = Model(inputs=[wav_in_clean, wav_in_noisy], outputs=[speech_D_out_T, T_G_wav, noise_D_out_T, T_G_noise_wav])
        G_S_D = Model(inputs=[wav_in_clean, wav_in_noisy], outputs=[speech_D_out_S, S_G_wav, noise_D_out_S, S_G_noise_wav])
    
    #G_T_D.compile(optimizer=g_opt,
    #          loss=['mean_squared_error', 'mean_absolute_error', 'mean_squared_error', 'mean_absolute_error'],
    #          loss_weights = [1, opts['g_l1loss'], 1, opts['g_l1loss']] )
    G_S_D.compile(optimizer=g_opt,
              loss=['mean_squared_error', 'mean_absolute_error', 'mean_squared_error', 'mean_absolute_error'],
              loss_weights = [1, opts['g_l1loss'], 1, opts['g_l1loss']] )
    
    #print ('G_T_D  : ',G_T_D.metrics_names)
    print ('G_S_D  : ',G_S_D.metrics_names)
    
    
                    
    #D_final.summary()
    #print (D_final.metrics_names)
    
    #exit ()
    
    if TEST_MODEL:
        ftestnoisy = h5py.File(noisy_test_matfile)
        noisy_test_data = ftestnoisy['feat_data']
        noisy_test_dfi = ftestnoisy['dfi']
        print ("Number of test files: " +  str(noisy_test_dfi.shape[1]) )


    # Begin the training part
    if TRAIN_MODEL:   
        fclean = h5py.File(clean_train_matfile)
        clean_train_data = np.array(fclean['feat_data']).astype('float32')
        #clean_train_data = clean_train_data[:,0:30000]
        #del fclean
        fnoisy = h5py.File(noisy_train_matfile)
        noisy_train_data = np.array(fnoisy['feat_data']).astype('float32')
        #noisy_train_data = noisy_train_data[:,0:30000]
        #del fnoisy
        numtrainsamples = clean_train_data.shape[1]
        idx_all = np.arange(numtrainsamples)
        # set random seed
        np.random.seed(opts['random_seed'])
        batch_size = opts['batch_size']

        print ("********************************************")
        print ("               SEGAN TRAINING               ")
        print ("********************************************")
        print ("Shape of clean feats mat " + str(clean_train_data.shape))
        print ("Shape of noisy feats mat " + str(noisy_train_data.shape))
        numtrainsamples = clean_train_data.shape[1]

        # Tensorboard stuff
        #log_path = './logs/T_' + modeldir
        #callback_t = TensorBoard(log_path)
        #callback_t.set_model(G_T_D)
        log_path = './logs/S_' + modeldir
        callback_s = TensorBoard(log_path)
        callback_s.set_model(G_S_D)
        train_names = ['G_loss', 'G_adv_loss', 'G_l1Loss', 'NG_adv_loss', 'NG_l1Loss']
    
        idx_all = np.arange(numtrainsamples)
        # set random seed
        np.random.seed(opts['random_seed'])
    
        batch_size = opts['batch_size']
        num_batches_per_epoch = int(np.floor(clean_train_data.shape[1]/batch_size))
        
        #for epoch in range(0):     
        #    t_g_loss_sum = 0      
        #    for batch_idx in range(num_batches_per_epoch):
        #        start_time = time.time()
        #        idx_beg = batch_idx * batch_size
        #        idx_end = idx_beg + batch_size
        #        idx = np.sort(np.array(idx_all[idx_beg:idx_end]))
        #        #print ("Batch idx " + str(idx[:5]) +" ... " + str(idx[-5:]))
        #        cleanwavs = np.array(clean_train_data[:,idx]).T
        #        cleanwavs = data_preprocess(cleanwavs, preemph=opts['preemph'])
        #        cleanwavs = np.expand_dims(cleanwavs, axis = 2)
        #        noisywavs = np.array(noisy_train_data[:,idx]).T
        #        noisywavs = data_preprocess(noisywavs, preemph=opts['preemph'])
        #        noisywavs = np.expand_dims(noisywavs, axis = 2)
        #        if not opts ['z_off']:
        #            noiseinput = np.random.normal(0, 1, (batch_size, z_dim1, z_dim2))
   # 
   #            if not opts['z_off']:
    #                t_g_loss = T_G.train_on_batch([noisywavs, noiseinput], cleanwavs)
    #                
    #            else:
    #                t_g_loss = T_G.train_on_batch([noisywavs], cleanwavs)
   # 
   #             t_g_loss_sum += t_g_loss
   #     
   #             time_taken = time.time() - start_time
   #         
   #             printlog = "E%d/%d:B%d/%d [T G loss: %f] [Exec. time: %f]" %  (epoch, n_epochs, batch_idx, num_batches_per_epoch, t_g_loss, time_taken)
   #             print (printlog)
                
    #        if min_ > t_g_loss_sum/num_batches_per_epoch:
    #            min_ = t_g_loss_sum/num_batches_per_epoch
    #            T_G.save_weights(modeldir + "/Gmodel_first.h5")
            
        #T_G.save_weights(modeldir + "/Gmodel_first.h5")     
        #S_G.load_weights('OLD_G_GAN_noZ_IN_Adam_D0.0002_G0.0002_L1_200.0/' + "/AGmodel_first_ep5.h5")  
        #S_G.load_weights('Weigths' + "/Gmodel_ep05.h5")     
        for epoch in range(n_epochs):
            g_l1_sum = 0
          
            # train D with  minibatch
            np.random.shuffle(idx_all) # shuffle the indices for the next epoch
            for batch_idx in range(num_batches_per_epoch):
                start_time = time.time()
                idx_beg = batch_idx * batch_size
                idx_end = idx_beg + batch_size
                idx = np.sort(np.array(idx_all[idx_beg:idx_end]))
                #print ("Batch idx " + str(idx[:5]) +" ... " + str(idx[-5:]))
                cleanwavs = np.array(clean_train_data[:,idx]).T
                cleanwavs = data_preprocess(cleanwavs, preemph=opts['preemph'])
                cleanwavs = np.expand_dims(cleanwavs, axis = 2)
                noisywavs = np.array(noisy_train_data[:,idx]).T
                noisywavs = data_preprocess(noisywavs, preemph=opts['preemph'])
                noisywavs = np.expand_dims(noisywavs, axis = 2)
                
                #if epoch%2==0:
                if not opts ['z_off']:
                    noiseinput = np.random.normal(0, 1, (batch_size, z_dim1, z_dim2))
                    g_out = S_G.predict([noisywavs, noiseinput])
                    noise_g_out = noisywavs - g_out
                else :
                    g_out = S_G.predict(noisywavs)
                    noise_g_out = noisywavs - g_out
                #else:
                #    if not opts ['z_off']:
                #        noiseinput = np.random.normal(0, 1, (batch_size, z_dim1, z_dim2))
                #        g_out = T_G.predict([noisywavs, noiseinput])
                #        noise_g_out = noisywavs - g_out
                #    else :
                #        g_out = T_G.predict(noisywavs)
                #        noise_g_out = noisywavs - g_out
                
                
                # train D     
              
                sd_real_loss = speech_D.train_on_batch ([cleanwavs,noisywavs], 
                                        opts ['D_real_target'] * np.ones((batch_size,1)))
                sd_fake_loss = speech_D.train_on_batch ([g_out,noisywavs], 
                                          np.zeros((batch_size,1)))
                sd_loss = 0.5 * np.add(sd_real_loss, sd_fake_loss)
    
                
                nd_real_loss = noise_D.train_on_batch ([cleanwavs,noisywavs], 
                                        opts ['D_real_target'] * np.ones((batch_size,1)))
                nd_fake_loss = noise_D.train_on_batch ([noise_g_out,noisywavs], 
                                          np.zeros((batch_size,1)))
                nd_loss = 0.5 * np.add(nd_real_loss, nd_fake_loss) 
    
                    
                # Train the combined model next; here, only the generator part is update
                valid_g = np.array([1]*(batch_size)) # generator wants discriminator to give 1 (identify fake as real)
               
                if not opts['z_off']:
                    [g_loss, g_dLoss, g_l1loss, n_gdLoss, n_gl1loss] = G_S_D.train_on_batch(
                                                            [cleanwavs, noisywavs, noiseinput], 
                                                            [valid_g, cleanwavs, valid_g, noisywavs-cleanwavs])
                        
                else:
                    [g_loss, g_dLoss, g_l1loss, n_gdLoss, n_gl1loss] = G_S_D.train_on_batch([cleanwavs, noisywavs], 
                                                            [valid_g, cleanwavs, valid_g, noisywavs-cleanwavs])
               
    
                time_taken = time.time() - start_time
    
                printlog = "E%d/%d:B%d/%d [SD loss: %f] [SD real: %f] [SD fake: %f] [ND loss: %f] [ND real: %f] [ND fake: %f] [G loss: %f] [G_D loss: %f] [G_L1 loss: %f] [NG_D loss: %f] [NG_L1 loss: %f] [Exec. time: %f]"  %  (epoch, n_epochs, batch_idx, num_batches_per_epoch, sd_loss, sd_real_loss, sd_fake_loss, nd_loss, nd_real_loss, nd_fake_loss, g_loss, g_dLoss, g_l1loss,  n_gdLoss, n_gl1loss, time_taken)
                
                g_l1_sum = g_l1_sum + g_l1loss
                print (printlog)
                # Tensorboard stuff 
                       
                logs = [g_loss, g_dLoss, g_l1loss, n_gdLoss, n_gl1loss]
                
                #if epoch%2 == 0:
                #write_log(callback_t, train_names, logs, epoch)
                #else:
                write_log(callback_s, train_names, logs, epoch)

            if g_l1_sum/num_batches_per_epoch < min_:
                min_ = g_loss
                #model_json = G.to_json()
                #with open("Gmodel.json", "w") as json_file:
                #    json_file.write(model_json)
                speech_D.save_weights(modeldir + "S_Dmodel_mi_ep0.h5")
                noise_D.save_weights(modeldir + "N_Dmodel_mi_ep0.h5")
                S_G.save_weights(modeldir + "S_Gmodel_mi_ep0.h5")
                #T_G.save_weights(modeldir + "T_Gmodel_mi_ep0.h5")

            if (TEST_MODEL and epoch % 10 == 0) or epoch == n_epochs - 1:
                print ("********************************************")
                print ("               SEGAN TESTING                ")
                print ("********************************************")

                resultsdir = modeldir + "/test_results_epoch" + str(epoch)
                if not os.path.exists(resultsdir):
                    os.makedirs(resultsdir)

                if LOAD_SAVED_MODEL:
                    print ("Loading model from " + modeldir + "/Gmodel_ep0")
                    json_file = open(modeldir + "/Gmodel_ep0.json", "r")
                    loaded_model_json = json_file.read()
                    json_file.close()
                    G_loaded = model_from_json(loaded_model_json)
                    G_loaded.compile(loss='mean_squared_error', optimizer=g_opt)
                    G_loaded.load_weights(modeldir + "/Gmodel_ep0.h5")
                else:
                    G_loaded = S_G

                print ("Saving Results to " + resultsdir)

                for test_num in tqdm(range(int(noisy_test_dfi.shape[1]))) :
                    test_beg = noisy_test_dfi[0, test_num]
                    test_end = noisy_test_dfi[1, test_num]
                    #print ("Reading indices " + str(test_beg) + " to " + str(test_end))
                    noisywavs = np.array(noisy_test_data[:,test_beg:test_end]).T
                    noisywavs = data_preprocess(noisywavs, preemph=opts['preemph'])
                    noisywavs = np.expand_dims(noisywavs, axis = 2)
                    if not opts['z_off']:
                        noiseinput = np.random.normal(0, 1, (noisywavs.shape[0], z_dim1, z_dim2))
                        cleaned_wavs = G_loaded.predict([noisywavs, noiseinput])
                    else :
                        cleaned_wavs = G_loaded.predict(noisywavs)
          
                    cleaned_wavs = np.reshape(cleaned_wavs, (noisywavs.shape[0], noisywavs.shape[1]))
                    cleanwav = reconstruct_wav(cleaned_wavs)
                    cleanwav = np.reshape(cleanwav, (-1,)) # make it to 1d by dropping the extra dimension
                    
                    if opts['preemph'] > 0:
                        cleanwav = de_emph(cleanwav, coeff=opts['preemph'])
                    
                    cleanwav = np.expand_dims(cleanwav, axis=1)
                    destfilename = resultsdir +  "/testwav_%d.wav" % (test_num)
                    wavfile.write(destfilename, fs, cleanwav/(np.max(np.abs(cleanwav))))
                    S_G.save_weights(resultsdir + "/S_Gmodel_ep0" + str(epoch)+".h5")
                    #T_G.save_weights(resultsdir + "/T_Gmodel_ep0" + str(epoch)+".h5")


        # Finally, save the model
        if SAVE_MODEL:
            model_json = S_G.to_json()
            with open(modeldir + "/S_Gmodel_ep0.json", "w") as json_file:
                json_file.write(model_json)
            S_G.save_weights(modeldir + "/S_Gmodel_final_ep0.h5")
            print ("Model saved to " + modeldir)
            
            #model_json = T_G.to_json()
            #with open(modeldir + "/T_Gmodel_ep0.json", "w") as json_file:
            #    json_file.write(model_json)
            #T_G.save_weights(modeldir + "/T_Gmodel_final_ep0.h5")
            #print ("Model saved to " + modeldir)
