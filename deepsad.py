import tensorflow as tf
from sklearn.metrics import roc_auc_score
from networks.mnist_LeNet import MNIST_LeNet

class DeepSAD():
    def __init__(self, cfg):
        
        self.eta = cfg['eta']
        self.c = None  
        self.eps = 1e-6

        self.model_name = cfg['model_name']
        self.model = None  

        self.trainer = None
        self.optimizer = None

        self.ae = None  
        self.ae_optimizer= None

        self.results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None,
            'test_scores': None,
        }

        self.ae_results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None
        }
        
    def build_model(self):
        
        implemented_networks = ('mnist_LeNet', 'mnist_DGM_M2', 'mnist_DGM_M1M2',
                            'fmnist_LeNet', 'fmnist_DGM_M2', 'fmnist_DGM_M1M2',
                            'cifar10_LeNet', 'cifar10_DGM_M2', 'cifar10_DGM_M1M2',
                            'arrhythmia_mlp', 'cardio_mlp', 'satellite_mlp', 'satimage-2_mlp', 'shuttle_mlp',
                            'thyroid_mlp',
                            'arrhythmia_DGM_M2', 'cardio_DGM_M2', 'satellite_DGM_M2', 'satimage-2_DGM_M2',
                            'shuttle_DGM_M2', 'thyroid_DGM_M2')
        assert self.model_name  in implemented_networks
       
        if self.model_name  == 'mnist_LeNet':
            self.model = MNIST_LeNet()
            
    def build_decoder(self, net_name):
        '''
        build decoder part which can match with model
        '''
        pass
#         if net_name == 'mnist_LeNet':
#             self.decoder =MNIST_LeNet()
    
    def init_c(self, eps = 0.1):
        print('Initialize center c')
        n = 0
        c = tf.zeros(self.model.rep_dim)
        for inputs, labels, semi in self.train_dataset:
            outputs = self.model(inputs)
            c+=tf.reduce_sum(outputs,0)
            n += inputs.shape[0]
        
        c /= n
        c = tf.where((c>=0)&(c<eps),eps,c)
        c = tf.where((c<0)&(c>-eps),-eps,c)

        self.c = c
   
    
    def train(self, train_dataset, lr, epochs, batch_size, beta1=0.9, beta2=0.999):
        
        self.train_dataset = train_dataset
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, beta_1=self.beta1, beta_2=self.beta2, epsilon=1e-08)
        
        self.init_c()
        for epoch in range(epochs):
            for inputs, _, semi_labels in self.train_dataset:
                loss = self.train_step(inputs, semi_labels)
            
            print(f"epoch: {epoch+1}/{epochs}, loss: {loss}")
            
        return
    
    def train_step(self, inputs, semi_labels):
        with tf.GradientTape() as tape:
            outputs = self.model(inputs, training=True)
            dist = tf.reduce_sum((outputs-self.c)**2,1)
            losses = tf.where(semi_labels == 0, dist, self.eta * ((dist + self.eps) ** tf.cast(semi_labels,dtype=tf.float32)))
            loss = tf.reduce_mean(losses)

        gradients = tape.gradient(loss,  self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients,  self.model.trainable_variables))
        
        return loss
        
    def test(self, test_dataset):
        
        print('Starting Test')
        label_list=[]
        score_list=[]
        
        for inputs, labels, semi_labels in self.train_dataset:
            outputs = self.model(inputs, training=False)
            dist = tf.reduce_sum((outputs-self.c)**2,1)
            label_list.append(labels)
            score_list.append(dist)
        
        labels = tf.concat(label_list,axis=0).numpy()
        scores = tf.concat(score_list,axis=0).numpy()
        self.test_auc = roc_auc_score(labels, scores)
        
        print('Test AUC: {:.2f}%'.format(100. * self.test_auc))