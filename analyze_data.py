import sys
import numpy as np
from collections import defaultdict
import tensorflow              #version '1.12.0'
tf = tensorflow.compat.v1      #if tf.__version__ >= 2.0
import time
tf.disable_eager_execution()
import pickle as pkl

n_folds = 10

assert len(sys.argv)==4,"usage: -NUM CLUSTERS, -RANDOM NUMBER SEED, -FOLD \in {1,...,5}"

K = int(sys.argv[1])
seed = int(sys.argv[2])
k = int(sys.argv[3])

f = open('main_params/trained_params_final_{}_{}_{}.pkl'.format(K,seed,k),'rb')
objects_to_pickle = pkl.load(f)
f.close()

LSTM_encoder_params,LSTM_decoder_params,params,log_theta = objects_to_pickle

def generate_data():
    text = [l.strip('\n').split('\t') for l in open('mackenzie_data_final.tsv','r')]
    MP_raw = [['<bos>']+list(l[0])+['<eos>'] for l in text]
    NP_raw = [['<bos>']+list(l[1])+['<eos>'] for l in text]
    input_segs = sorted(set([s for w in MP_raw for s in w]))
    output_segs = sorted(set([s for w in NP_raw for s in w]))
    S_i = len(input_segs)
    S_o = len(output_segs)
    T_i = max([len(l) for l in MP_raw])
    T_o = max([len(l) for l in NP_raw])
    N = len(MP_raw)
    MP = np.ones([N,T_i],dtype=np.int32)*-1
    for i,w in enumerate(MP_raw):
        for j,s in enumerate(w):
            MP[i,j] = input_segs.index(s)
    NP = np.ones([N,T_o],dtype=np.int32)*-1
    for i,w in enumerate(NP_raw):
        for j,s in enumerate(w):
            NP[i,j] = output_segs.index(s)
    encoder_input = MP
    decoder_input = NP[:,:-1]
    decoder_output = NP[:,1:]
    T_o -= 1
    return(MP_raw,NP_raw,input_segs,output_segs,S_i,S_o,T_i,T_o,N,MP,NP,encoder_input,decoder_input,decoder_output)

MP_raw,NP_raw,input_segs,output_segs,S_i,S_o,T_i,T_o,N,MP,NP,encoder_input,decoder_input,decoder_output = generate_data()

np.random.seed(seed)

all_idx = np.arange(N)
np.random.shuffle(all_idx)

fold_idx = list(zip(list(range(0,N,int(N/n_folds)+1)),list(range(int(N/n_folds)+1,N,int(N/n_folds)+1))+[N]))

train_idx = np.array([i for i in all_idx if i not in range(fold_idx[k][0],fold_idx[k][1])])
test_idx = np.array([i for i in all_idx if i in range(fold_idx[k][0],fold_idx[k][1])])

N_ = len(train_idx)

D = 64

batch_idx = tf.placeholder(tf.int32,shape=(None,),name='batch_idx')

struc_zeros = tf.expand_dims(tf.cast(np.triu(np.ones([T_i,T_i])),dtype='float32'),0)

def MultiLSTM(inputs,weights):
    w_kernel = weights['w_kernel']
    w_recurrent = weights['w_recurrent']
    w_bias = weights['w_bias']
    T = inputs.shape[-2]
    H = []
    for t in range(T):
        if t > 0:
            z = tf.einsum('nx,kxj->nkj',inputs[:,t,:],w_kernel) + tf.einsum('nkl,klj->nkj',h,w_recurrent) + tf.expand_dims(w_bias,0)
        else:
            z = tf.einsum('nx,kxj->nkj',inputs[:,t,:],w_kernel) + tf.expand_dims(w_bias,0)
        i,f,o,u = tf.split(z,4,axis=-1)
        i = tf.sigmoid(i)        #input gate
        f = tf.sigmoid(f + 1.0)  #forget gate
        o = tf.sigmoid(o)        #output gate
        u = tf.tanh(u)           #information let in by input gate
        if t > 0:
            c = f * c + i * u
        else:
            c = i * u
        h = o * tf.tanh(c)
        H.append(h)
    H = tf.stack(H,-2)
    return(H)

def MultiBiLSTM(inputs,weights_fwd,weights_bkwd):
    """birectional LSTM"""
    forward = MultiLSTM(inputs,weights_fwd)
    backward = tf.reverse(MultiLSTM(tf.reverse(inputs,[-2]),weights_bkwd),[-2])
    return(tf.concat([forward,backward],-1))

def EncoderDecoder(encoder_input,decoder_input):
    h_enc = MultiBiLSTM(encoder_input,LSTM_encoder_params['enc_fwd'],LSTM_encoder_params['enc_bkwd'])
    h_dec = MultiLSTM(decoder_input,LSTM_decoder_params['dec_fwd'])
    #alpha = tf.exp(tf.einsum('nktj,nksj->nkst',h_dec,tf.einsum('klj,nksj->nksl',params['T'],h_enc)))
    alpha = tf.nn.softmax(tf.einsum('nktj,nksj->nkst',h_dec,tf.einsum('klj,nksj->nksl',params['T'],h_enc)),-2)
    alignment_probs = []
    for i in range(T_o):
        if i == 0:
            alpha_prev = alpha[:,:,:,i]/tf.reduce_sum(alpha[:,:,:,i],-1,keepdims=True)
        if i > 0:
            alpha_prev = tf.einsum('nkx,nky->nkxy',alpha[:,:,:,i],alignment_probs[i-1])
            alpha_prev *= struc_zeros
            alpha_prev = tf.reduce_sum(alpha_prev,-2)+1e-6
            alpha_prev /= tf.reduce_sum(alpha_prev,-1,keepdims=True)
        alignment_probs.append(alpha_prev)
    alignment_probs = tf.stack(alignment_probs,-1)
    h_enc_rep = tf.tile(tf.expand_dims(h_enc,-2),[1,1,1,T_o,1])
    h_dec_rep = tf.tile(tf.expand_dims(h_dec,-3),[1,1,T_i,1,1])
    h_rep = tf.concat([h_enc_rep,h_dec_rep],-1)
    emission_probs = tf.nn.tanh(tf.einsum('kjl,nkstj->nkstl',params['V'],h_rep))
    emission_probs = tf.nn.softmax(tf.einsum('kyj,nkstj->nksty',params['W'],emission_probs))
    prob_out = tf.expand_dims(alignment_probs,-1)*emission_probs
    pred_out = tf.reduce_sum(prob_out,-3)
    return(pred_out)

def model():
    encoder_input_ = tf.one_hot(tf.gather(encoder_input,batch_idx),S_i)
    decoder_input_ = tf.one_hot(tf.gather(decoder_input,batch_idx),S_o)
    decoder_output_ = tf.one_hot(tf.gather(decoder_output,batch_idx),S_o)
    log_p_out_z = tf.log(EncoderDecoder(encoder_input_,decoder_input_))
    log_lik_z = tf.expand_dims(decoder_output_,1)*log_p_out_z
    log_lik_z = tf.reduce_sum(tf.reduce_sum(log_lik_z,-1),-1) + tf.nn.log_softmax(tf.expand_dims(log_theta,-2))
    log_lik = tf.reduce_logsumexp(log_lik_z,-1)
    nlloss = -tf.reduce_mean(log_lik)
    return(log_p_out_z,log_lik_z,nlloss)

batch_size = 32

model_predict,model_eval,model_nlloss = model()

sess = tf.Session()

log_p_clust = sess.run(model_eval,feed_dict={batch_idx:test_idx})

z = np.argmax(log_p_clust,1)

print(log_theta)

#print('\n'.join([' '.join(MP_raw[i])+' > '+' '.join(NP_raw[i])+' '+str(z[j]) #+
#          #' '+'|'.join([str(s) for s in log_p_clust[j]]) 
#          for j,i in enumerate(test_idx)]))