import numpy as np
from tensorflow import keras

class qnetwork():

  def __init__(self, env, gamma=1.0, lr=None):
    n_actions = env.action_space.n
    state_shape = env.observation_space.shape

    state = keras.Input(shape=state_shape, name='state')
    x = keras.layers.Dense(256, activation='relu', name='hidden_1')(state)
    x = keras.layers.Dense(128, activation='relu', name='hidden_2')(x)
    Q = keras.layers.Dense(n_actions, activation='linear', name='Q')(x)
    qmodel = keras.Model(inputs=state, outputs=Q)

    target_qmodel = keras.models.model_from_config({
      'class_name': qmodel.__class__.__name__, 
      'config': qmodel.get_config()
    })

    mask = keras.Input(shape=(n_actions,), name='action_mask')    
    def project(args):
      pred_qs, mask = args
      return keras.backend.batch_dot(pred_qs, mask)
    out = keras.layers.Lambda(project, output_shape=(1,), name='project')([Q, mask])
  
    trainable_qmodel = keras.Model(inputs=[state, mask], outputs=out)
    if lr is not None:
      optimizer = keras.optimizers.Adam(learning_rate=lr)
    else:
      optimizer = keras.optimizers.Adam()
    trainable_qmodel.compile(optimizer=optimizer, loss='mean_squared_error')

    self.n_actions = n_actions
    self.state_shape = state_shape
    self.qmodel = qmodel
    self.target_qmodel = target_qmodel
    self.trainable_qmodel = trainable_qmodel
    self.gamma = gamma

  # state, action, reward, next_state, done should all have the same shape[0]
  def train(self, state, action, reward, next_state, done):
    Q_next = np.max(self.target_qmodel.predict(next_state), axis=1)
    target_Q = reward + self.gamma * Q_next * (1 - done)
    action_mask = keras.utils.to_categorical(action, self.n_actions)

    return self.trainable_qmodel.train_on_batch([state, action_mask], target_Q)

  def update_target_model(self):
    self.target_qmodel.set_weights(self.qmodel.get_weights())

  # Given a batch of states, return the actions
  def predict(self, state):
    return self.qmodel.predict(state)

  # state should be a single state, not batched
  def get_best_action(self, state):
    return np.argmax(self.qmodel.predict(state[np.newaxis, :]))

  def save(self, path):
    self.qmodel.save(path)
