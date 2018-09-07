from GAN import GAN
import tensorflow as tf


class WGAN(GAN):

    def losses(self, D_f_logits, D_r_logits):
        """parameters: D_f_logits, D_r_logits
        returns:(G_loss, D_loss)
        """       
        D_loss = -(tf.reduce_mean(D_r_logits) - tf.reduce_mean(D_f_logits))
        G_loss = -tf.reduce_mean(D_f_logits) 
        return(G_loss, D_loss)

    def optimize(self, G_loss, D_loss, G_optimizer, D_optimizer, clip=0.01):
        """parameters: G_loss, D_loss, G_optimizer, D_optimizer, clip=0.01
        returns:(G_opt, D_opt_grouped)
        """         
        G_Vars, D_Vars = self._get_vars()

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):

            G_opt = G_optimizer.minimize(G_loss, var_list=G_Vars)
            D_opt = D_optimizer.minimize(D_loss, var_list=D_Vars)             
            with tf.control_dependencies([D_opt]):
                D_ops_list = list()
                for d_var in D_Vars:
                    D_ops_list.append(tf.assign(d_var, tf.clip_by_value(d_var, -clip, clip)))
                D_opt_grouped = tf.group(D_ops_list)    
        return(G_opt, D_opt_grouped)
            
    def build_graph(self, G_layers, D_layers, x, z, add_sum, opt_type, clip, lr, use_decay):
        """parameters: G_layers, D_layers, x, z, add_sum, opt_type, clip, lr, use_decay
        returns:(graph, x, z, is_train, G_col, G_out, D_r_col, D_r_logits, D_f_col, D_f_logits, sums_img, sums_loss, G_opt, D_opt)
        """                   
       
        is_train = tf.placeholder(dtype=tf.bool)
    
        G_col, G_out = self.gan_block("generator", z, G_layers, is_train, False)
    
        D_r_col, D_r_logits = self.gan_block("discriminator", x, D_layers, is_train, False)
        D_f_col, D_f_logits = self.gan_block("discriminator", G_out, D_layers, is_train, True)
        assert(G_out.shape.as_list()==x.shape.as_list())
    
        #lr = 2e-4
    
        G_loss, D_loss = self.losses(D_f_logits, D_r_logits)
        sums_img, sums_loss = None, None
        if add_sum:
            sums_img = tf.summary.merge([tf.summary.image("real_images", x[0:2, ...]), 
                                         tf.summary.image("fake_images", G_out[0:2, ...])])
            sums_loss = tf.summary.merge([tf.summary.scalar("D_loss", D_loss), tf.summary.scalar("G_loss", G_loss)])
            
        if use_decay:
            lr_D = tf.train.exponential_decay(lr, tf.Variable(0, trainable=False), 1000, 0.95, True)
            lr_G = tf.train.exponential_decay(lr, tf.Variable(0, trainable=False), 1000, 0.95, True)
        else:
            lr_D = lr
            lr_G = lr
        
        if opt_type == "RMS":
            G_optimizer = tf.train.RMSPropOptimizer(lr_G)
            D_optimizer = tf.train.RMSPropOptimizer(lr_D)
        elif opt_type == "Adam":
            G_optimizer = tf.train.AdamOptimizer(lr_G)
            D_optimizer = tf.train.AdamOptimizer(lr_D)
            
        G_opt, D_opt = self.optimize(G_loss, D_loss, G_optimizer, D_optimizer, clip)
        init = tf.global_variables_initializer()
        graph = tf.get_default_graph()
        
        return(graph, x, z, is_train, G_col, G_out, D_r_col, D_r_logits, D_f_col, D_f_logits, sums_img, sums_loss, D_loss, G_loss, G_opt, D_opt, init)
    
    def train(self, batch_size, epochs, c_epochs, rep_cyc, get_batch, z_type, add_sum, sum_path, save_path,  gargs):
        """parameters: batch_size, epochs, get_batch, z_type, add_sum, sum_path, save_path,  **gargs
        """             
        graph, x, z, is_train, G_col, G_out, D_r_col, D_r_logits, D_f_col, D_f_logits, sums_img, sums_loss, D_loss, G_loss, G_opt, D_opt, init = gargs

        print("Training Started!")
        
        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(init)
            if add_sum:
                writer = tf.summary.FileWriter(logdir=sum_path)

            cnt = 0
            for epoch in range(epochs):

                real_image = get_batch(batch_size)
                #real_image = (real_image - 0.5)/0.5
                #Train discriminator
                for _ in range(c_epochs):
                    z_val = self._get_z_code(batch_size, z_type)
    
                    feed_dict={z:z_val, x:real_image, is_train:True}
                    D_loss_val, _ = sess.run([D_loss, D_opt], feed_dict=feed_dict)

                #Train generator
                z_val = self._get_z_code(batch_size, z_type)
                feed_dict={z:z_val, x:real_image, is_train:True}
                G_loss_val, _ = sess.run([G_loss, G_opt], feed_dict=feed_dict)

                if cnt%rep_cyc==0:
                    if add_sum:
                        writer.add_summary(sess.run(sums_loss, feed_dict=feed_dict), global_step=cnt)
                    print("\r epoch {} cnt {} Loss discriminator {}   Loss generator {}".
                          format(epoch, cnt, D_loss_val, G_loss_val), end=" ")
                    z_val = self._get_z_code(batch_size, z_type)
                    feed_dict={z:z_val, x:real_image, is_train:False}
                    if add_sum:
                        writer.add_summary(sess.run(sums_img, feed_dict=feed_dict), global_step=cnt)

                cnt += 1
            saver = tf.train.Saver(var_list=tf.global_variables())
            saver.save(sess, save_path, global_step=epoch) 
        print("Training Finished model saved!")       
            
        

class GP_WGAN(WGAN):
    def __init__(self, verbose=True):
        self._verbose = verbose
    def losses(self, D_f_logits, D_r_logits, D_n_logits, lamb_da, batch_size):
        """parameters: D_f_logits, D_r_logits, D_n_logits, lamb_da, batch_size
        returns:(G_loss, D_loss)
        """          

        _, D_Vars = self._get_vars()
        d = tf.reshape(D_n_logits, [-1, 1])
        ds = tf.split(d, axis=0, num_or_size_splits=batch_size)
        gs = [tf.gradients(ys=l, xs=w) for l, w in zip(ds, [D_Vars for _ in ds])]
        gs = [tf.concat([tf.reshape(g, [-1]) for g in gl], axis=0) for gl in gs]
        gs = tf.stack(gs)

        gs =tf.square(tf.norm(gs, axis=1) - 1)
        D_loss = tf.reduce_mean(D_f_logits) - tf.reduce_mean(D_r_logits) + lamb_da*tf.reduce_mean(gs)
        G_loss = -tf.reduce_mean(D_f_logits)
        return(G_loss, D_loss)    
    
    
    def build_graph(self, G_layers, D_layers, x, z, add_sum, opt_type, lamb_da, lr, use_decay):
        """parameters: G_layers, D_layers, x, z, add_sum, opt_type, clip, lr, use_decay
        returns:(graph, x, z, is_train, G_col, G_out, D_r_col, D_r_logits, D_f_col, D_f_logits, sums_img, sums_loss, G_opt, D_opt)
        """                   
       
        is_train = tf.placeholder(dtype=tf.bool)
    
        G_col, G_out = self.gan_block("generator", z, G_layers, is_train, False)
        batch_size = x.shape.as_list()[0]
        e = tf.random_uniform([batch_size, 1, 1, 1], 0, 1)
        x_n = e*x + (1 - e)*G_out        
    
        D_r_col, D_r_logits = self.gan_block("discriminator", x, D_layers, is_train, False)
        D_f_col, D_f_logits = self.gan_block("discriminator", G_out, D_layers, is_train, True)
        D_n_col, D_n_logits = self.gan_block("discriminator", x_n, D_layers, is_train, True)
       
        assert(G_out.shape.as_list()==x.shape.as_list())
    
        #lr = 2e-4
    
        G_loss, D_loss = self.losses(D_f_logits, D_r_logits, D_n_logits, lamb_da, batch_size)
        
        sums_img, sums_loss = None, None
        if add_sum:
            sums_img = tf.summary.merge([tf.summary.image("real_images", x[0:2, ...]), 
                                         tf.summary.image("fake_images", G_out[0:2, ...])])
            sums_loss = tf.summary.merge([tf.summary.scalar("D_loss", D_loss), tf.summary.scalar("G_loss", G_loss)])
            
        if use_decay:
            lr_D = tf.train.exponential_decay(lr, tf.Variable(0, trainable=False), 1000, 0.95, True)
            lr_G = tf.train.exponential_decay(lr, tf.Variable(0, trainable=False), 1000, 0.95, True)
        else:
            lr_D = lr
            lr_G = lr
        
        if opt_type == "RMS":
            G_optimizer = tf.train.RMSPropOptimizer(lr_G)
            D_optimizer = tf.train.RMSPropOptimizer(lr_D)
        elif opt_type == "Adam":
            G_optimizer = tf.train.AdamOptimizer(lr_G)
            D_optimizer = tf.train.AdamOptimizer(lr_D)
            
        G_opt, D_opt = GAN.optimize(self, G_loss, D_loss, G_optimizer, D_optimizer)
        init = tf.global_variables_initializer()
        graph = tf.get_default_graph()
        
        return(graph, x, z, is_train, G_col, G_out, D_r_col, D_r_logits, D_f_col, D_f_logits, sums_img, sums_loss, D_loss, G_loss, G_opt, D_opt, init)
    
       
    
    
    
def run_tests():
    print("Starting test")
    tf.reset_default_graph()
 
    ####################################################
    ####################W GAN tests####################
    ####################################################
    tf.reset_default_graph()  
    gan = WGAN()
    G_layers=[[1024, [4, 4], (1, 1), "valid", tf.random_normal_initializer(), 1, "r", None, 0],
              [512, [4, 4], (2, 2), "same", tf.random_normal_initializer(), 1, "r", None, 1],
              [256, [4, 4], (2, 2), "same", tf.random_normal_initializer(), 0, "th", None, 0],
              [128, [4, 4], (2, 2), "same", None, 0, "s", None, 0],
              [1, [4, 4], (2, 2), "same", None, 0, "th", None, 0]]  
    
    D_layers=[[128, [4, 4], (2, 2), "same", tf.random_normal_initializer(), 0, "lr", None, 0],
              [256, [4, 4], (2, 2), "same", tf.random_normal_initializer(), 1, "lr", None, 0],
              [512, [4, 4], (2, 2), "same", tf.random_normal_initializer(), 1, "lr", None, 0],
              [1024, [4, 4], (2, 2), "valid", tf.random_normal_initializer(), 0, "l", None, 0]] 
    
    batch_size = 60
    x = tf.placeholder(shape=[batch_size, 64, 64, 1], dtype=tf.float32)
    z = tf.placeholder(shape=[batch_size, 1, 1, 100], dtype=tf.float32)
               
    _, out_g = gan.gan_block("generator", z, G_layers, True, False)
    assert(x.shape.as_list()==out_g.shape.as_list())
      
    _, out_d_r = gan.gan_block("discriminator", x, D_layers , True, False)
    _, out_d_f = gan.gan_block("discriminator", out_g, D_layers, True, True)

    assert(x.shape.as_list()==out_g.shape.as_list())
    loss = gan.losses(out_d_f, out_d_r)
    
    assert(len(loss)==2)
    assert(loss[0].shape.as_list()==[])
    assert(loss[1].shape.as_list()==[])    
    print("W_GAN tests Passed")
    
    
    ####################################################
    ####################GP GAN tests####################
    ####################################################
    
    tf.reset_default_graph()  
    gan = GP_WGAN()
    G_layers=[[1024, [4, 4], (1, 1), "valid", tf.random_normal_initializer(), 1, "r", None, 0],
              [512, [4, 4], (2, 2), "same", tf.random_normal_initializer(), 1, "r", None, 0],
              [256, [4, 4], (2, 2), "same", tf.random_normal_initializer(), 0, "th", None, 1],
              [128, [4, 4], (2, 2), "same", None, 0, "s", None, 0],
              [1, [4, 4], (2, 2), "same", None, 0, "th", None, 0]]  
    
    D_layers=[[128, [4, 4], (2, 2), "same", tf.random_normal_initializer(), 0, "lr", None, 0],
              [256, [4, 4], (2, 2), "same", tf.random_normal_initializer(), 1, "lr", None, 0],
              [512, [4, 4], (2, 2), "same", tf.random_normal_initializer(), 1, "lr", None, 0],
              [1024, [4, 4], (2, 2), "valid", tf.random_normal_initializer(), 0, "l", None, 0]] 
    
    batch_size = 60
    x = tf.placeholder(shape=[batch_size, 64, 64, 1], dtype=tf.float32)
    e = tf.random_uniform([batch_size, 1, 1, 1], 0, 1)
    z = tf.placeholder(shape=[batch_size, 1, 1, 100], dtype=tf.float32)
    
    
                       
    _, out_g = gan.gan_block("generator", z, G_layers, True, False)
    assert(x.shape.as_list()==out_g.shape.as_list())
    x_n = e*x + (1 - e)*out_g
    
    _, out_d_r = gan.gan_block("discriminator", x, D_layers , True, False)
    _, out_d_f = gan.gan_block("discriminator", out_g, D_layers, True, True)
    print(x.shape.as_list(), out_g.shape.as_list(), x_n.shape.as_list())
    assert(x.shape.as_list()==out_g.shape.as_list()==x_n.shape.as_list())
    _, out_d_n = gan.gan_block("discriminator", x_n , D_layers, True, True)
    loss = gan.losses(out_d_f, out_d_r, out_d_n, 0.1, batch_size)
    
    assert(len(loss)==2)
    assert(loss[0].shape.as_list()==[])
    assert(loss[1].shape.as_list()==[])
    print("GP GAN tests Passed!")
    tf.reset_default_graph()
    
    print("All test Passed!")

if __name__=="__main__":
    
    run_tests()