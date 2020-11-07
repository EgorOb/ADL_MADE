import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from tqdm.notebook import tqdm
import numpy as np

class QLerning():
    '''
    Обучение поведения агента алгоритмом QL
    '''
    def __init__(self, model, lf=0.5, df=0.5):
#         Модель поведения агента
        self._model = model
#       Матрица полезности для агента определенная в модели
        self._matrix_q = model.get_matrix_q()
#       Матрица полезности выдающая максимальное среднее значение выигрыша
        self._best_matrix_q = self._matrix_q.copy()
#       Фактор обучения. Чем он выше, тем сильнее агент доверяет новой информации.
        self.lf = lf
#       Фактор дисконтирования. Чем он меньше, тем меньше агент задумывается о выгоде от будущих своих действий.
        self.df = df
    
        self._mean_reward = -1000
        self._mean_reward_from_iter = -1000
        
        self._game_is_done = False
        self._last_state = self._model._min_state
        self._last_action = 0
        self._last_reward = 0

    def _choice_action(self, current_state):
        '''Выбор действия по матрице полезности от текущего состояния'''
        row = self._model.convert_state_action_to_index_matrix_q(current_state, 0)[0]
        return np.argmax(self._matrix_q[row,:])
        
    def _update_matrix_q(self,):
        '''Обновление матрицы полезности для предыдущего шага'''
        current_index = self._model.convert_state_action_to_index_matrix_q(self._current_state, 0)
        last_index = self._model.convert_state_action_to_index_matrix_q(self._last_state, self._last_action)
        
        self._max_q = max(self._matrix_q[current_index[0], :])
        self._matrix_q[last_index] += self.lf * (self._last_reward + self.df * self._max_q - self._matrix_q[last_index])    
    
    def _update_state_action(self,):
        """
        Обновление параметров состояний и действий
            Метод get_new_state(action) в модели поведения возвращает значения параметров 
            после совершения действия над средой:
                - значение текущего состояния агента;
                - награду за переход в текущее состояние агента;
                - флаг определяющий состояние игры (True - игра закончена, False - игра продолжается)
        """
#         Переопределение значений для предыдущего шага
        self._last_state = self._current_state
        self._last_action = self._current_action
        
#         Определение значений для текущего шага
        new_state, reward, self._game_is_done = self._model.get_new_state_from_lerning(self._current_state, self._last_action)
        if self._game_is_done == False:
            self._current_state = new_state
            self._current_action = self._model.choice_action(self._current_state)
        self._last_reward = reward
             
    def _one_game_traning(self,):
        """Обучение на одной игре"""
        self._game_is_done = False
        self._model.reset_game()
        self._current_state = self._model.get_current_state()
        self._current_action = self._model.choice_action(self._current_state)

        while self._game_is_done == False:            
            self._update_matrix_q()
            self._update_state_action()
    
    def traning_model(self, 
                      n_train = 1000,
                      show_fit = False,
                      delta_show = 10, 
                      n_games = 10000):
        
        """
        Обучение для n_train игр
            n_train - число итераций для тренировки модели поведения в среде
            show_fit - флаг отвечающий за отображение изменения среднего выигрыша в процессе обучения
            delta_show - через сколько итераций n_train показывать изменение среднего выигрыша в процессе обучения
            n_games - число игр, для подсчета среднего выигрыша
        """
        if show_fit:
            self._fig, self._ax = plt.subplots()
            list_mean_rewards = []
            list_i = []
            plt.ion()
        
        for i in tqdm(range(n_train)):
            self._one_game_traning()
#             Обработчик события отрисовки
            if show_fit and i%delta_show == 0:
                self._model.game(matrix_q=self._matrix_q, n_games=n_games)
                self._mean_reward = self._model.get_mean_reward()
#                 Обновление информации о лучшей матрице полезности
                if self._mean_reward > self._mean_reward_from_iter:
                    self._mean_reward_from_iter = self._mean_reward
                    print("value from best iter =", self._mean_reward_from_iter)
                    self._best_matrix_q = self._matrix_q.copy()
                
                list_mean_rewards.append(self._mean_reward)
                list_i.append(i)
#                 Отрисовка среднего выигрыша
                self._draw_plot(list_i, list_mean_rewards)
        
        if show_fit:
            print("best value all time=", self._mean_reward_from_iter)
            plt.ioff() 
    
    def get_mean_reward(self,):
        '''Вывод текущего среднего выигрыша'''
        return self._mean_reward
    
    def get_matrix_q(self,):
        '''Вывод текущей матрицы полезности'''
        return self._matrix_q

    def get_best_mean_reward(self,):
        '''Вывод лучшего среднего выигрыша'''
        return self._mean_reward_from_iter
    
    def get_best_matrix_q(self,):
        '''Вывод лучшей матрицы полезности'''
        return self._best_matrix_q
    
    def get_matrix_actions(self,):
        """Вывод матрицы действий в зависимости от состояний"""
        return np.argmax(self._best_matrix_q, axis=1)
    
    def _draw_plot(self, data_x, data_y):
        '''Отрисовка значений среднего выигрыша'''
        self._ax.clear()
        self._ax.plot(data_x, data_y)
        self._ax.set_title('Зависимость среднего выигрыша от итерации обучения')
        self._ax.set_ylabel('Средний выигрыш')
        self._ax.set_xlabel('Итерация обучения')
        self._fig.canvas.draw()