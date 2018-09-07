from GAN import GAN
import tensorflow as tf

class LSGAN(GAN):
    def losses(self, D_f_logits, D_r_logits, smoothing_factor=None):
        """parameters: D_f_logits, D_r_logits, , smoothing_factor=None)
        returns:(G_loss, D_loss)
        """     
        D_loss = 0.5*tf.reduce_mean(tf.square(D_r_logits - tf.ones_like(D_r_logits))) + 0.5*tf.reduce_mean(tf.square(D_f_logits))
        G_loss = 0.5*tf.reduce_mean(tf.square(D_f_logits - tf.ones_like(D_f_logits)))
        return(G_loss, D_loss)
    
