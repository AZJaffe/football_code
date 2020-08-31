import sys
sys.path.append('/Users/ajaffe/Dropbox/research/football/code/sfm')
import trainsfm
import datetime

time = datetime.datetime.now().strftime('%H:%M:%S')
trainsfm.train(
  lr=0.001, 
  batch_size=16,
  conv_depth = 2,
  num_hidden_layers=2,
  fc_layer_width=512,
  data_dir='../../datasets/gradient_ball_slow_speed_40x24',
  tensorboard_dir=f'./run{time}',
  sliding_data=False,
  num_epochs=20,
  vis_freq = 20,
  n_vis_point=5,
  forwbackwreg_coeff=0.001,
  displreg_coeff=0.01,
  maskreg_coeff=0.01
)