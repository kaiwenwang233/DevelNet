import os
import random
import threading
import tensorflow as tf
import numpy as np
import pandas as pd
import logging
import scipy as sp
import scipy.io
pd.options.mode.chained_assignment = None


class Config():
  seed = 100
  use_seed = False
  n_channel = 1
  n_class = 2
  num_repeat_noise = 1
  sampling_rate = 100
  dt = 1.0/sampling_rate
  X_shape = [252, 387, n_channel]
  Y_shape = [1, 387, n_class]
  min_event_gap = 3 * sampling_rate


class DataReader(object):

  def __init__(self,
               data_dir,
               data_list,
               mask_window,
               queue_size,
               coord,
               config=Config()):
    self.config = config
    tmp_list = pd.read_csv(data_list, header=0)
    self.data_list = tmp_list
    self.num_data = len(self.data_list)
    self.data_dir = data_dir
    self.queue_size = queue_size
    self.n_channel = config.n_channel
    self.n_class = config.n_class
    self.X_shape = config.X_shape
    self.Y_shape = config.Y_shape
    self.min_event_gap = config.min_event_gap
    self.mask_window = int(mask_window * config.sampling_rate)
    self.coord = coord
    self.threads = []
    self.add_placeholder()
  
  def add_placeholder(self):
    self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=self.config.X_shape)
    self.target_placeholder = tf.placeholder(dtype=tf.float32, shape=self.config.Y_shape)
    self.queue = tf.PaddingFIFOQueue(self.queue_size,
                                     ['float32', 'float32'],
                                     shapes=[self.config.X_shape, self.config.Y_shape])
    self.enqueue = self.queue.enqueue([self.sample_placeholder, self.target_placeholder])

  def dequeue(self, num_elements):
    output = self.queue.dequeue_many(num_elements)
    # output = self.queue.dequeue_up_to(num_elements)
    return output

  def normalize(self, data):
    data -= np.mean(data)
    # std_data = np.std(data, axis=0, keepdims=True)
    # assert(std_data.shape[-1] == data.shape[-1])
    # std_data[std_data == 0] = 1
    data /= np.std(data)
    return data

  def scale_amplitude(self, data):
    if random.uniform(0, 1) < 0.2:
      data *= random.uniform(1, 2)
    elif random.uniform(0, 1) < 0.4:
      data /= random.uniform(1, 10)
    return data

  def drop_channel(self, data):
    if random.uniform(0, 1) < 0.3:
      c1 = random.choice([0, 1])
      c2 = random.choice([0, 1])
      c3 = random.choice([0, 1, 1, 1])
      if c1 + c2 + c3 > 0:
        data[..., np.array([c1, c2, c3]) == 0] = 0
        # data *= 3/(c1+c2+c3)
    return data

  def add_noise(self, data, channels):
    while random.uniform(0, 1) < 0.1:
      meta = np.load(os.path.join(self.data_dir, (self.data_list[self.data_list['channels']==channels]).sample(n=1).iloc[0]['fname']))
      data += self.normalize(meta['data'][:self.X_shape[0], np.newaxis, :]) * random.uniform(1, 3)
    return data

  def adjust_amplitude_for_multichannels(self, data):
    tmp = np.max(np.abs(data), axis=0, keepdims=True)
    assert(tmp.shape[-1] == data.shape[-1])
    if np.count_nonzero(tmp) > 0:
      data *= data.shape[-1] / np.count_nonzero(tmp)
    return data

  def add_event(self, data, itp_list, its_list, channels, normalize=False):
    while random.uniform(0, 1) < 0.1:
    # for i in range(3):
      shift = None
      meta = np.load(os.path.join(
          self.data_dir, (self.data_list[self.data_list['channels']==channels]).sample(n=1).iloc[0]['fname']))
      start_tp = meta['itp'].tolist()
      itp = meta['itp'].tolist() - start_tp
      its = meta['its'].tolist() - start_tp

      # shift = random.randint(-self.X_shape[1], self.X_shape[1])
      if (max(its_list) - itp + self.mask_window + self.min_event_gap > self.X_shape[0]-self.mask_window) \
         and (its - min(itp_list) + self.mask_window + self.min_event_gap > min([its, self.X_shape[0]]) - self.mask_window):
        # return data, itp_list, its_list
        continue
      elif max(its_list) - itp + self.mask_window + self.min_event_gap > self.X_shape[0]-self.mask_window:
        shift = random.randint(its - min(itp_list)+self.mask_window + self.min_event_gap, min([its, self.X_shape[0]])-self.mask_window)
      elif its - min(itp_list) + self.mask_window + self.min_event_gap > min([its, self.X_shape[0]]) - self.mask_window:
        shift = -random.randint(max(its_list) - itp + self.mask_window + self.min_event_gap, self.X_shape[0] - self.mask_window)
      else:
        shift = random.choice([-random.randint(max(its_list) - itp + self.mask_window + self.min_event_gap, self.X_shape[0] - self.mask_window), 
                               random.randint(its - min(itp_list)+self.mask_window + self.min_event_gap, min([its, self.X_shape[0]])-self.mask_window)])
      if normalize:
        data += self.normalize(meta['data'][start_tp+shift:start_tp+self.X_shape[0]+shift, np.newaxis, :])
      else:
        data += meta['data'][start_tp+shift:start_tp+self.X_shape[0]+shift, np.newaxis, :]
      itp_list.append(itp-shift)
      its_list.append(its-shift)
    return data, itp_list, its_list

  def thread_main(self, sess, n_threads=1, start=0):
    stop = False
    # Go through the dataset multiple times
    while not stop:
      index = list(range(start, self.num_data, n_threads))
      random.shuffle(index)
      for i in index:
        fname = os.path.join(self.data_dir, self.data_list.iloc[i]['fname'])
        if os.path.exists(fname):
          meta = sp.io.loadmat(fname)
        #data = 1-self.normalize(np.squeeze(meta['synImage']))
        data = 1-np.squeeze(meta['synImage'])
        # channels = meta['channels'].tolist()
        # start_tp = meta['itp'].tolist()

        if self.coord.should_stop():
          stop = True
          break

        sample = np.zeros(self.X_shape)
        sample[:,:,:] = data[:,:, np.newaxis]
        # if self.config.use_seed:
        #   np.random.seed(self.config.seed+i)
        # if np.random.random() < 0.9:
          # shift = random.randint(-(self.X_shape[0]-self.mask_window), min([meta['its'].tolist()-start_tp, self.X_shape[0]])-self.mask_window)
          # sample[:, :, :] = data[start_tp+shift:start_tp+self.X_shape[0]+shift, np.newaxis, :]
          # itp_list = [meta['itp'].tolist()-start_tp-shift]
          # its_list = [meta['its'].tolist()-start_tp-shift]

          # data augmentation
          # sample = self.normalize(sample)
          # sample, itp_list, its_list = self.add_event(sample, itp_list, its_list, channels, normalize=True)
          # sample = self.add_noise(sample)
          # sample = self.scale_amplitude(sample)
          # if len(channels.split('_')) == 3:
          #   sample = self.drop_channel(sample)
        # else:  # pure noise
        #   sample[:, :, :] = data[start_tp-self.X_shape[0]:start_tp, np.newaxis, :]
        #   itp_list = []
        #   its_list = []

        # sample = self.normalize(sample)
        # sample = self.adjust_amplitude_for_multichannels(sample)

        if (np.isnan(sample).any() or np.isinf(sample).any() or (not sample.any())):
          continue

        target = np.zeros(self.Y_shape)
#        istart = int(np.min(meta['pslabel'][:,0:2]))
#        iend = int(np.max(meta['pslabel'][:,0:2]))
#        target[0,istart:iend,1] = 1  
        for i in range(meta['pslabel'].shape[1]//2):
          istart = int(np.min(meta['pslabel'][:,2*i:2*i+2]))
          iend = int(np.max(meta['pslabel'][:,2*i:2*i+2]))
          target[0,np.max([istart,0]):np.min([iend,387]),1] = 1  
        target[:,:,0] = 1 - target[:,:,1]
        # for itp, its in zip(itp_list, its_list):
        #   if (itp-self.mask_window//2 >= target.shape[0]) or (itp+self.mask_window//2 < 0):
        #     pass
        #   elif (itp-self.mask_window//2 >= 0) and (itp-self.mask_window//2 < target.shape[0]):
        #     target[itp-self.mask_window//2:itp+self.mask_window//2, 0, 1] = np.exp(-(np.arange(
        #            itp-self.mask_window//2,itp+self.mask_window//2)-itp)**2/(2*(self.mask_window//4)**2))[:target.shape[0]-(itp-self.mask_window//2)]
        #   elif (itp-self.mask_window//2 < target.shape[0]):
        #     target[0:itp+self.mask_window//2, 0, 1] = np.exp(-(np.arange(
        #            0,itp+self.mask_window//2)-itp)**2/(2*(self.mask_window//4)**2))[:target.shape[0]-(itp-self.mask_window//2)]
        #   if (its-self.mask_window//2 >= target.shape[0]) or (its+self.mask_window//2 < 0):
        #     pass
        #   elif (its-self.mask_window//2 >= 0) and (its-self.mask_window//2 < target.shape[0]):
        #     target[its-self.mask_window//2:its+self.mask_window//2, 0, 2] = np.exp(-(np.arange(
        #            its-self.mask_window//2,its+self.mask_window//2)-its)**2/(2*(self.mask_window//4)**2))[:target.shape[0]-(its-self.mask_window//2)]
        #   elif (its-self.mask_window//2 < target.shape[0]):
        #     target[0:its+self.mask_window//2, 0, 2] = np.exp(-(np.arange(
        #            0,its+self.mask_window//2)-its)**2/(2*(self.mask_window//4)**2))[:target.shape[0]-(its-self.mask_window//2)]
        # target[:, :, 0] = 1 - target[:, :, 1] - target[:, :, 2]

        sess.run(self.enqueue, feed_dict={self.sample_placeholder: sample,
                                          self.target_placeholder: target})
    return 0

  def start_threads(self, sess, n_threads=8):
    for i in range(n_threads):
      thread = threading.Thread(target=self.thread_main, args=(sess, n_threads, i))
      thread.daemon = True  # Thread will close when parent quits.
      thread.start()
      self.threads.append(thread)
    return self.threads


class DataReader_valid(DataReader):

  def add_placeholder(self):
    self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
    self.target_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
    self.fname_placeholder = tf.placeholder(dtype=tf.string, shape=None)
    self.itp_placeholder = tf.placeholder(dtype=tf.int32, shape=None)
    self.its_placeholder = tf.placeholder(dtype=tf.int32, shape=None)
    self.queue = tf.PaddingFIFOQueue(self.queue_size,
                                     ['float32', 'float32', 'string', 'int32', 'int32'],
                                     shapes=[self.config.X_shape, self.config.Y_shape, [], [None], [None]])
    self.enqueue = self.queue.enqueue([self.sample_placeholder, self.target_placeholder, 
                                       self.fname_placeholder, 
                                       self.itp_placeholder, self.its_placeholder])

  def dequeue(self, num_elements):
    # output = self.queue.dequeue_many(num_elements)
    output = self.queue.dequeue_up_to(num_elements)
    return output

  def thread_main(self, sess, n_threads=1, start=0):
    index = list(range(start, self.num_data, n_threads))
    for i in index:
      fname = os.path.join(self.data_dir, self.data_list.iloc[i]['fname'])
      meta = np.load(fname)
      data = np.squeeze(meta['data'])
      channels = meta['channels'].tolist()
      start_tp = meta['itp'].tolist()
      
      if self.coord.should_stop():
        break

      sample = np.zeros(self.X_shape)
      if self.config.use_seed:
        np.random.seed(self.config.seed+i)
      if np.random.random() < 0.9:
        shift = random.randint(-(self.X_shape[0]-self.mask_window), min([meta['its'].tolist()-start_tp, self.X_shape[0]])-self.mask_window)
        sample[:, :, :] = data[start_tp+shift:start_tp+self.X_shape[0]+shift, np.newaxis, :]
        itp_list = [meta['itp'].tolist()-start_tp-shift]
        its_list = [meta['its'].tolist()-start_tp-shift]

        # data augmentation
        sample = self.normalize(sample)
        sample, itp_list, its_list = self.add_event(sample, itp_list, its_list, channels, normalize=True)
        # sample = self.add_noise(sample)
        # sample = self.scale_amplitude(sample)
        if len(channels.split('_')) == 3:
          sample = self.drop_channel(sample)
      else:  # pure noise
        sample[:, :, :] = data[start_tp-self.X_shape[0]:start_tp, np.newaxis, :]
        itp_list = []
        its_list = []

      sample = self.normalize(sample)
      sample = self.adjust_amplitude_for_multichannels(sample)

      if (np.isnan(sample).any() or np.isinf(sample).any() or (not sample.any())):
        continue

      target = np.zeros(self.Y_shape)
      itp_true = []
      its_true = []
      for itp, its in zip(itp_list, its_list):
        if (itp-self.mask_window//2 >= target.shape[0]) or (itp+self.mask_window//2 < 0):
          pass
        elif (itp-self.mask_window//2 >= 0) and (itp-self.mask_window//2 < target.shape[0]):
          target[itp-self.mask_window//2:itp+self.mask_window//2, 0, 1] = np.exp(-(np.arange(
                  itp-self.mask_window//2,itp+self.mask_window//2)-itp)**2/(2*(self.mask_window//4)**2))[:target.shape[0]-(itp-self.mask_window//2)]
          itp_true.append(itp)
        elif (itp-self.mask_window//2 < target.shape[0]):
          target[0:itp+self.mask_window//2, 0, 1] = np.exp(-(np.arange(
                  0,itp+self.mask_window//2)-itp)**2/(2*(self.mask_window//4)**2))[:target.shape[0]-(itp-self.mask_window//2)]
          itp_true.append(itp)

        if (its-self.mask_window//2 >= target.shape[0]) or (its+self.mask_window//2 < 0):
          pass
        elif (its-self.mask_window//2 >= 0) and (its-self.mask_window//2 < target.shape[0]):
          target[its-self.mask_window//2:its+self.mask_window//2, 0, 2] = np.exp(-(np.arange(
                  its-self.mask_window//2,its+self.mask_window//2)-its)**2/(2*(self.mask_window//4)**2))[:target.shape[0]-(its-self.mask_window//2)]
          its_true.append(its)
        elif (its-self.mask_window//2 < target.shape[0]):
          target[0:its+self.mask_window//2, 0, 2] = np.exp(-(np.arange(
                  0,its+self.mask_window//2)-its)**2/(2*(self.mask_window//4)**2))[:target.shape[0]-(its-self.mask_window//2)]
          its_true.append(its)
      target[:, :, 0] = 1 - target[:, :, 1] - target[:, :, 2]

      sess.run(self.enqueue, feed_dict={self.sample_placeholder: sample,
                                        self.target_placeholder: target,
                                        self.fname_placeholder: fname,
                                        self.itp_placeholder: itp_true,
                                        self.its_placeholder: its_true})
    return 0


class DataReader_pred(DataReader):

  def __init__(self,
               data_dir,
               data_list,
               queue_size,
               coord,
               input_length=None,
               config=Config()):
    self.config = config
    tmp_list = pd.read_csv(data_list, header=0)
    self.data_list = tmp_list
    self.num_data = len(self.data_list)
    self.data_dir = data_dir
    self.queue_size = queue_size
    self.X_shape = config.X_shape
    self.Y_shape = config.Y_shape
    if input_length is not None:
      logging.warning("Using input length: {}".format(input_length))
      self.X_shape[1] = input_length
      self.Y_shape[1] = input_length

    self.coord = coord
    self.threads = []
    self.add_placeholder()
  
  def add_placeholder(self):
    self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
    self.fname_placeholder = tf.placeholder(dtype=tf.string, shape=None)
    self.queue = tf.PaddingFIFOQueue(self.queue_size,
                                     ['float32', 'string'],
                                     shapes=[self.config.X_shape, []])
    self.enqueue = self.queue.enqueue([self.sample_placeholder, self.fname_placeholder])

  def dequeue(self, num_elements):
    # output = self.queue.dequeue_many(num_elements)
    output = self.queue.dequeue_up_to(num_elements)
    return output

  def thread_main(self, sess, n_threads=1, start=0):
    index = list(range(start, self.num_data, n_threads))
    for i in index:
      fname = self.data_list.iloc[i]['fname']
      try:
        meta = sp.io.loadmat(os.path.join(self.data_dir, fname))
      except:
        logging.warning("Loading {} failed!".format(fname))
      #data = self.normalize(meta['synImage'])
      data = 1-np.squeeze(meta['synImage'])
      #data = self.normalize(meta['synImage'])
      sample = np.zeros(self.X_shape)
      sample[:,:,:] = data[:, :, np.newaxis]
      
      if np.array(sample.shape).all() != np.array(self.X_shape).all():
        logging.error("{}: shape {} is not same as input shape {}!".format(fname, sample.shape, self.X_shape))
        continue

      if np.isnan(sample).any() or np.isinf(sample).any():
        logging.warning("Data error: {}\nReplacing nan and inf with zeros".format(fname))
        sample[np.isnan(sample)] = 0
        sample[np.isinf(sample)] = 0

      # sample = self.normalize(sample)
      # sample = self.adjust_amplitude_for_multichannels(sample)
      sess.run(self.enqueue, feed_dict={self.sample_placeholder: sample,
                                        self.fname_placeholder: fname})


if __name__ == "__main__":
  pass

