import tensorflow as tf
import numpy as np



class GAN:
    
    def __init__(self, verbose=True):
        self._verbose = verbose
    
    def _get_minibatch_discr(self, input_, T):
        
        M = tf.matmul(input_, tf.reshape(T, [T.shape.as_list()[0], -1]))
        M = tf.reshape(M, [-1, T.shape.as_list()[1], T.shape.as_list()[2]])
        M_r = tf.expand_dims(M, 1)
        M_r_T = tf.transpose(M_r, perm=[1, 0, 2, 3])
        
        # - is to get rid of the self minus
        c = tf.exp(-tf.norm(M_r - M_r_T, ord=1, axis=3))
        o = tf.reduce_sum(c, axis=1) - 1
        return(M, c, o)    
    def _get_vars(self):
        G_Vars = [g_var for g_var in tf.trainable_variables() if g_var.name.startswith("generator")]
        D_Vars = [d_var for d_var in tf.trainable_variables() if d_var.name.startswith("discriminator")]
        
        assert(len(set(tf.trainable_variables()).difference(set(G_Vars).union(set(D_Vars)))) == 0)  
        return(G_Vars, D_Vars)
    
    def _get_z_code(self, batch_size, z_type, z_dim):
        if z_type=="n":
            return(np.random.normal(0, 1, size=[batch_size, 1, 1, z_dim]))
        if z_type=="u":
            return(np.random.uniform(-0.5, 0.5, size=[batch_size, 1, 1, z_dim]))    
        
    
    def gan_block(self, l_type, input_, layers, is_train, reuse):
        """parameters: l_type, input_, layers, is_train, reuse
        layers=[[1024, [4, 4], (1, 1), "valid", tf.random_normal_initializer(), 1, "lr|r|l|s|th|sp", <[mb_filter_add, mb_M_col]>None|[4, 32]], <add_noise>0|1]
        returns:(ep_collection, out)
        """
        do_print = self._verbose 
        
        #G_layers=[[1024, [4, 4], (1, 1), "valid", tf.random_normal_initializer(), 1, "lr|r|l|s|th|sp", <[mb_filter_add, mb_M_col]>None|[4, 32]], <add_noise>0|0.01]
        assert(l_type == "discriminator" or l_type == "generator")
        
        with tf.variable_scope(name_or_scope=l_type, reuse=reuse):
            out = input_
            ep_collection = list()
            if do_print:
                print("{}{}{}".format("="*5, l_type, "="*5))
            
            for n, [no_of_filters, k_size, strides, padding, initializer, has_bn, act, mb, noise] in enumerate(layers):
                end_points = list()
                assert(((noise is None or noise == 0) and l_type=="generator") or l_type=="discriminator")
                
                if not(noise is None or noise == 0) and l_type=="discriminator":
                    out =  tf.add(out, tf.random_normal(shape=tf.shape(out), stddev=noise))
                        
                if l_type == "generator":
                    out = tf.layers.conv2d_transpose(out, no_of_filters, k_size, strides, padding, kernel_initializer=initializer, name="conv_tp_{}".format(n)) 
                elif l_type == "discriminator":
                    out = tf.layers.conv2d(out, no_of_filters, k_size, strides, padding, kernel_initializer=initializer, name="conv_{}".format(n))
                else:
                    raise Exception("Wrong GAN block type")
                end_points.append(out)
                
                if not(has_bn is None or has_bn != 1):
                    out = tf.layers.batch_normalization(out, name="bn_{}".format(n), training=is_train)
                    end_points.append(out)
                    
                    
                if act=="lr":
                    out = tf.nn.leaky_relu(out, alpha=0.2, name="lrelu_{}".format(n))
                elif act=="r":
                    out = tf.nn.relu(out, name="relu")
                elif act=="l":
                    out = tf.identity(out, "linear")
                elif act=="s":
                    out = tf.nn.sigmoid(out, name="sig")
                elif act=="th":
                    out = tf.nn.tanh(out, name="tanh")
                elif act=="sp":
                    out = tf.nn.softplus(out, name="softplus")
                else:
                    raise NameError
                       
                    
                end_points.append(out)
                if l_type=="discriminator" and mb is not None:
                    assert(len(mb)==2)
                    out_shape = out.shape.as_list()
                    out_flat = tf.reshape(out, [-1,out_shape[1]*out_shape[2]*out_shape[3]])
                                  
                    out_flat_shape = out_flat.shape.as_list()
                    T = tf.get_variable(shape=[out_flat_shape[-1], out_shape[1]*out_shape[2]*mb[0], mb[1]], name="T_{}".format(n))
                    _ ,_ , o = self._get_minibatch_discr(out_flat, T)
                    out = tf.reshape(tf.concat([out_flat, o], axis=1), [-1, out_shape[1], out_shape[2], out_shape[3]+mb[0]])
                layer_cfg = "layer:{}, no_of_filters:{}, k_size:{}, strides:{}, padding:{}, initializer:{}, has_bn:{}, act:{} output shape:{} mb:{} noise:{}".\
                                                      format(n, no_of_filters, k_size, strides, padding, initializer, has_bn, act, out.shape.as_list(), mb, noise)             
                if do_print:
                    print("-"*(len(layer_cfg)+2))
                    print("-{}-".format(layer_cfg))
                ep_collection.append(end_points)
                                 
            return(ep_collection, out)
        
    def losses(self, D_f_logits, D_r_logits, smoothing_factor=1.0):
        """parameters: D_f_logits, D_r_logits, smoothing_factor=1.0)
        returns:(G_loss, D_loss)
        """ 
                 
        G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_f_logits, labels=tf.ones_like(D_f_logits)))
        D_r_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_r_logits, labels=tf.ones_like(D_r_logits)*smoothing_factor)) 
        D_f_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_f_logits, labels=tf.zeros_like(D_f_logits)))
        D_loss = D_r_loss + D_f_loss
        return(G_loss, D_loss)
    
      
    def optimize(self, G_loss, D_loss, G_optimizer, D_optimizer):
        """parameters: G_loss, D_loss, G_optimizer, D_optimizer
        returns:(G_opt, D_opt)
        """              
        G_Vars, D_Vars = self._get_vars()
            
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            G_opt = G_optimizer.minimize(G_loss, var_list=G_Vars)
            D_opt = D_optimizer.minimize(D_loss, var_list=D_Vars) 
        return(G_opt, D_opt)
    
    def build_graph(self, G_layers, D_layers, x, z, add_sum, smoothing, opt_type, lr, use_decay):
        """parameters: G_layers, D_layers, x, z, add_sum, smoothing, opt_type[type, 0.9, 0.99], lr, use_decay
        layers=[[1024, [4, 4], (1, 1), "valid", tf.random_normal_initializer(), 1, "lr|r|l|s|th|sp", <[mb_filter_add, mb_M_col]>None|[4, 32]], <add_noise>0|0.01]
        returns:(graph, x, z, is_train, G_col, G_out, D_r_col, D_r_logits, D_f_col, D_f_logits, sums_img, sums_loss, G_opt, D_opt)
        """                   
       
        is_train = tf.placeholder(dtype=tf.bool)
    
        G_col, G_out = self.gan_block("generator", z, G_layers, is_train, False)
    
        D_r_col, D_r_logits = self.gan_block("discriminator", x, D_layers, is_train, False)
        D_f_col, D_f_logits = self.gan_block("discriminator", G_out, D_layers, is_train, True)
        assert(G_out.shape.as_list()==x.shape.as_list())
    
        #lr = 2e-4
    
        G_loss, D_loss = self.losses(D_f_logits, D_r_logits, smoothing_factor=smoothing)
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
        
        if opt_type[0] == "RMS":
            G_optimizer = tf.train.RMSPropOptimizer(lr_G)
            D_optimizer = tf.train.RMSPropOptimizer(lr_D)
        elif opt_type[0] == "Adam":
            G_optimizer = tf.train.AdamOptimizer(lr_G, beta1=opt_type[1], beta2=opt_type[2])
            D_optimizer = tf.train.AdamOptimizer(lr_D, beta1=opt_type[1], beta2=opt_type[2])
                
            
        G_opt, D_opt = self.optimize(G_loss, D_loss, G_optimizer, D_optimizer)
        init = tf.global_variables_initializer()
        graph = tf.get_default_graph()
        
        return(graph, x, z, is_train, G_col, G_out, D_r_col, D_r_logits, D_f_col, D_f_logits, sums_img, sums_loss, D_loss, G_loss, G_opt, D_opt, init)
    
    def build_inference_graph(self, G_layers, z, save_path):
        """parameters: G_layers
        layers=[[1024, [4, 4], (1, 1), "valid", tf.random_normal_initializer(), 1, "lr|r|l|s|th|sp", <[mb_filter_add, mb_M_col]>None|[4, 32]]]
        returns:(graph, x, z, is_train, G_col, G_out, D_r_col, D_r_logits, D_f_col, D_f_logits, sums_img, sums_loss, G_opt, D_opt)
        """                   
    
    
        is_train = tf.constant(0)
        G_col, G_out = self.gan_block("generator", z, G_layers, is_train, False)
        graph = tf.get_default_graph()
        sess = tf.Session(graph=graph)
        if save_path.endswith(".npz") or save_path.endswith(".npy"):
            load_obj = np.load(save_path)
            file_key = load_obj.files[0]
            vars_dict = load_obj[file_key].item(0)
            for v in graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                sess.run(v.assign(vars_dict[v.name]))
        else:

            saver = tf.train.Saver()
            saver.restore(sess, save_path)
    
        return(graph, sess, z, G_col, G_out)
    
    
    def train(self, batch_size, epochs, rep_cyc, get_batch, z_type, z_dim, add_sum, sum_path, save_path, restore_path, save_checkpoints, gargs):
        """parameters: batch_size, epochs, get_batch, z_type, add_sum, sum_path, save_path,  **gargs
        """             
        graph, x, z, is_train, G_col, G_out, D_r_col, D_r_logits, D_f_col, D_f_logits, sums_img, sums_loss, D_loss, G_loss, G_opt, D_opt, init = gargs

        print("Training Started!")
        
        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            saver = tf.train.Saver(var_list=tf.global_variables())
            
            if not (restore_path == None):
                saver.restore(sess, restore_path)
            else:
                sess.run(init)
            if add_sum:
                writer = tf.summary.FileWriter(logdir=sum_path)

            cnt = 0
            for epoch in range(epochs):

                real_image = get_batch(batch_size)
                #real_image = (real_image - 0.5)/0.5
                #Train discriminator
                z_val = self._get_z_code(batch_size, z_type, z_dim)

                feed_dict={z:z_val, x:real_image, is_train:True}
                D_loss_val, _ = sess.run([D_loss, D_opt], feed_dict=feed_dict)

                #Train generator
                z_val = self._get_z_code(batch_size, z_type, z_dim)
                feed_dict={z:z_val, x:real_image, is_train:True}
                G_loss_val, _ = sess.run([G_loss, G_opt], feed_dict=feed_dict)

                if cnt%rep_cyc==0:
                    if add_sum:
                        writer.add_summary(sess.run(sums_loss, feed_dict=feed_dict), global_step=cnt)
                    print("\r epoch {} cnt {} Loss discriminator {}   Loss generator {}".
                          format(epoch, cnt, D_loss_val, G_loss_val), end=" ")
                    z_val = self._get_z_code(batch_size, z_type, z_dim)
                    feed_dict={z:z_val, x:real_image, is_train:False}
                    if add_sum:
                        writer.add_summary(sess.run(sums_img, feed_dict=feed_dict), global_step=cnt)

                if not (save_checkpoints==None):
                    for cp in save_checkpoints:
                        if cnt==cp and cnt!=epochs:
                            saver.save(sess, save_path, global_step=cnt)                             
                            
                cnt += 1
 
            saver.save(sess, save_path, global_step=epoch) 
        print("Training Finished model saved!")        
       
  
    
def run_tests():
    print("Starting test")
    tf.reset_default_graph()
    gan = GAN()
    G_layers=[[1024, [4, 4], (1, 1), "valid", tf.random_normal_initializer(), 1, "lr", None, 0],
              [512, [4, 4], (2, 2), "same", tf.random_normal_initializer(), 0, "r", "salam", 0],
              [256, [4, 4], (2, 2), "same", tf.random_normal_initializer(), 1, "sp", None, 0],
              [128, [4, 4], (2, 2), "same", None, 0, "s", None, 0],
              [1, [4, 4], (2, 2), "same", None, 0, "th", None, 0]]
    
    ep_col, out = gan.gan_block("generator", tf.placeholder(shape=[None, 1, 1, 100], dtype=tf.float32), G_layers, True, False)
    assert(ep_col[0][-1].shape.as_list() == [None, 4, 4, 1024])
    assert("conv_tp" in ep_col[0][0].name)
    assert("BatchNorm" in ep_col[0][1].name)
    
    assert("lrelu" in ep_col[0][-1].name)
    assert("softplus" in ep_col[2][-1].name)
    
    assert(ep_col[1][-1].shape.as_list() == [None, 8, 8, 512])
    assert("BatchNorm" not in ep_col[1][1].name)
    assert("relu" in ep_col[1][-1].name)
    
    assert("sig" in ep_col[3][-1].name) 
    assert("tanh" in ep_col[4][-1].name) 
    
    tf.reset_default_graph()
    gan = GAN()
    
    D_layers=[[128, [4, 4], (2, 2), "same", tf.random_normal_initializer(), 1, "lr", None, 0],
              [256, [4, 4], (2, 2), "same", tf.random_normal_initializer(), 0, "r", None, 1],
              [512, [4, 4], (2, 2), "same", tf.random_normal_initializer(), 1, "l", [4, 16], 0],
              [1024, [4, 4], (2, 2), "valid", tf.random_normal_initializer(), 0, "s", None,0]]
    
    ep_col, out = gan.gan_block("discriminator", tf.placeholder(shape=[None, 64, 64, 1], dtype=tf.float32), D_layers, True, False)
    assert(ep_col[0][-1].shape.as_list() == [None, 32, 32, 128])
    assert("conv" in ep_col[0][0].name)
    assert("BatchNorm" in ep_col[0][1].name)
    assert("lrelu" in ep_col[0][-1].name)
    
    assert(ep_col[1][-1].shape.as_list() == [None, 16, 16, 256])
    assert("BatchNorm" not in ep_col[1][1].name)
    assert("relu" in ep_col[1][-1].name)
    assert("sig" in ep_col[3][-1].name)
    assert('discriminator/T:0' in [f.name for f in tf.trainable_variables()])

    
    try:
        gan.gan_block("discriminator", tf.placeholder(shape=[None, 64, 64, 1], dtype=tf.float32), D_layers, True, False)
        assert(False)
    except ValueError:
        assert(True)
        
    try:
        gan.gan_block("discriminator", tf.placeholder(shape=[None, 64, 64, 1], dtype=tf.float32), D_layers, True, True)
        assert(True)
    except ValueError:
        assert(False)

    #testing minibatch functaion
    np.random.seed(9)
    in_val = np.random.uniform(size=[5, 10])
    input_ = tf.constant(in_val)
    
    np.random.seed(1)
    T_val = np.random.uniform(size=[10, 8, 12])
    T = tf.constant(T_val)
    M, c, o = gan._get_minibatch_discr(input_, T)
    
    assert(M.shape.as_list() == [5, 8, 12])
    assert(c.shape.as_list() == [5, 5, 8])
    sess = tf.InteractiveSession()
    assert(o.shape.as_list() == [5, 8])
    
    M_test = np.dot(in_val, T_val.reshape([10, -1])).reshape([5, 8, 12])
    
    assert(np.max(M.eval()[3, ...] - M_test[3, ...])<1e-15)
    c_test = np.ndarray([4])
    for idx, i in enumerate([0, 1, 2, 4]):
        c_test[idx] = np.exp(-np.sum(np.abs(M_test[3, 2,...] - M_test[i, 2, ...])))
    
    assert(np.sum(c_test) - o.eval()[3,2]<1e-15)
    
    print("GAN tests Passed!")
    
  
    print("All test Passed!")

if __name__=="__main__":
    
    run_tests()

    
                                                 
                                                 
            
                               
        
        
        
            
