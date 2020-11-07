import gym
import numpy as np
import multiprocessing as mp
from time import time
from tqdm.notebook import tqdm

class ModelUtils:
    '''
    Методы для моделей игр в блэкджэк
    '''
    def get_matrix_q(self,):
        """Возвращение текущей матрицы полезности"""
        return self._matrix_q
    
    def get_mean_reward(self,):
        """Возвращение среднего выигрыша"""
        return self._mean_reward   
    
    def replace_matrix_q(self, new_matrix_q):
        """
        Замена текущей матрицы полезности
            new_matrix_q - матрица для замены
        """
        size_new = new_matrix_q.shape
        size_old = self._matrix_q.shape
        assert size_new == size_old, 'Размеры матриц не совпадают'
        self._matrix_q = new_matrix_q
    
    def convert_state_action_to_index_matrix_q(self, state, action):
        """
        Конвертация состояния и действия в позицию строки и столбца матрицы полезности.
        """
        return state, action
    
    '''
    Ниже приведены функции взятые из доп.функций класса блэкджек
    '''
    
    def cmp(self, a, b):
        return float(a > b) - float(a < b)
    
    def draw_card(self, deck):
        '''deck - объект класса колоды карт'''
        return int(deck.get_card_from_deck(1))
    
    def draw_hand(self, deck):
        '''deck - объект класса колоды карт'''
        return list(deck.get_card_from_deck(2))

    def usable_ace(self, hand):  # Does this hand have a usable ace?
        return 1 in hand and sum(hand) + 10 <= 21

    def sum_hand(self, hand):  # Return current hand total
        if self.usable_ace(hand):
            return sum(hand) + 10
        return sum(hand)

    def is_bust(self, hand):  # Is this hand a bust?
        return self.sum_hand(hand) > 21

    def score(self, hand):  # What is the score of this hand (0 if bust)
        return 0 if self.is_bust(hand) else self.sum_hand(hand)

    def is_natural(self, hand):  # Is this hand a natural blackjack?
        return sorted(hand) == [1, 10]
    

class Deck():
    """
    Класс определяющий объект колоды и операций над ней
        shufle_treshold - порог для перемешивания колоды. 
            Для перемещивания колоды воспользоваться методом shufle()
            Если в колоде карт больше, чем shufle_treshold, то перемещивание не произодет
        counts_shuffle - число раз перемещивания колоды
        model_score - модель подсчета карт
            "plus_minus" - модель "плюс-минус"
            "half" - модель "половинки"
    """
    def __init__(self, shufle_treshold=15, counts_shuffle=1, model_score="plus_minus"):
        self.deck = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]*4)
        self._card_drop_out = 0
        self._shufle_treshold = shufle_treshold
        self._counts_shuffle = counts_shuffle
#         np.random.seed(int(seed))
        self.shufle()
        
        self._model_score = model_score
        self._model_score_cards = {"plus_minus":np.array([-1, 1, 1, 1, 1, 1, 0, 0, 0, -1]),
                                   "half":np.array([-1, 0.5, 1, 1, 1.5, 1, 0.5, 0, -0.5, -1])}
#         "plus_minus" - модель "плюс-минус"
#         "half" - модель "половинки"
        self._deck_score = 0
        
    def shufle(self):
        '''Перемешивание карт в колоде'''
        for i in range(self._counts_shuffle):
            np.random.shuffle(self.deck)
        self._card_drop_out = 0
        self._deck_score = 0
        
    def need_shufle(self,):
        '''Проверка необходимости перемешивать колоду'''
        if self._card_drop_out >= 52 - self._shufle_treshold:
            return True
        return False
    
    def get_card_from_deck(self, counts_card):
        '''Получение необходимого числа карт из колоды'''
        cards = self.deck[self._card_drop_out : self._card_drop_out + counts_card]
        self._card_drop_out += counts_card
        self._deck_score += self.solve_score(cards - 1)
        return cards
    
    def get_balance(self,):
        '''Вернуть оставшееся число карт в колоде'''
        return int(52 - self._card_drop_out)
    
    def get_model_score_cards(self):
        '''Получение матрицы баллов карт в модели'''
        return self._model_score_cards[self._model_score]
    
    def get_score_deck(self,):
        '''Получение текущих очков колоды'''
        return self._deck_score
    
    def solve_score(self, cards):
        '''
        Расчет суммы баллов карт
            cards - лист карт для подсчета очков
        '''
        if len(cards)>0:
            return sum(self._model_score_cards[self._model_score][cards])
        else:
            return 0
        
class SimpleModel(ModelUtils):
    '''
    Простая модель игры в блэкджэк
    '''
    def __init__(self, natural=True, matrix_q=None, matrix_reward=None, use_matrix_reward=False):
        self._env = gym.make('Blackjack-v0', natural=natural)
        self._env1 = gym.make('Blackjack-v0', natural=natural)
#       Матрица полезности действий
        if isinstance(matrix_q, np.ndarray):
            size = matrix_q.shape
            assert size == (18, 2), 'Размер матрицы полезности должен быть 18x2'
            self._matrix_q = matrix_q
        else:
            self._matrix_q = np.random.rand(18,2)
#             self._matrix_q = np.ones((18,2))
#         Минимально возможное значнение карт на руках
        self._min_state = 4
        self._mean_reward = 0
        self._current_state = self._min_state
        
        self._use_matrix_reward = use_matrix_reward
        if self._use_matrix_reward:
            if isinstance(matrix_reward, np.ndarray):
                size = matrix_reward.shape
                assert size == (18, 2), 'Размер матрицы награды должен быть 18x2'
                self._matrix_reward = matrix_reward
            else:
                self._solve_matrix_reward()
    
    def game(self, n_games = 10000, matrix_q=None):
        '''Инициализации n_games партий'''
        if isinstance(matrix_q, np.ndarray):
             self.replace_matrix_q(matrix_q)
        sum_reward = 0
        for i in range(n_games):
            self.reset_game()
            done = False
            while done == False:
                action = self.choice_action(self._current_state)
                new_state, reward, done = self.get_new_state(action)
                if done == False:
                    self._current_state = new_state
            sum_reward += reward    
        self._mean_reward = sum_reward / n_games
    
    def reset_game(self,):
        """"""
        state = self._env.reset()
        self._current_state = state[0]
        
    def get_current_state(self,):
        return self._current_state
   
    def choice_action(self, current_state):
        '''
        Выбор действия по матрице полезности от текущего состояния
        current_state - текущее состояние
        '''
        row = self.convert_state_action_to_index_matrix_q(current_state, 0)[0]
        return np.argmax(self._matrix_q[row,:])
     
    def get_new_state(self, action):
        '''Возвращение нового состояния, награды и флага состояния игры после действия'''
        state, reward, done, _ = self._env.step(action)
        return state[0], reward, done
    
    def _get_reward(self, state, action, n_games=10000):
        sum_ = 0
        for i in range(n_games):
            self._env1.reset()
            self._env1.player = [state]
            st, rew, done, _ = self._env1.step(action)
            sum_ += rew
        return sum_/n_games
    
    def _solve_matrix_reward(self):
        print("Расчёт матрицы ожидания награды")
        self._matrix_reward = np.zeros_like(self._matrix_q)
        for i in tqdm(range(self._matrix_reward.shape[0])):
            for j in range(self._matrix_reward.shape[1]):
                self._matrix_reward[i,j] = self._get_reward(i + 4, j)
        
    def get_matrix_reward(self):
        return self._matrix_reward
    
    def get_new_state_from_lerning(self, current_state, action):
        '''Возвращение нового состояния, награды и флага состояния игры после действия'''
        if self._use_matrix_reward:
            state, reward, done, _ = self._env.step(action)
            reward = self._matrix_reward[self.convert_state_action_to_index_matrix_q(current_state, action)]
            return state[0], reward, done
        else:
            return self.get_new_state(action)
        
    def convert_state_action_to_index_matrix_q(self, state, action):
        """Конвертация состояния и действия в позицию строки и столбца матрицы полезности."""
        return state - self._min_state, action

class DoubleModel(ModelUtils):
    '''
    Простая модель игры в блэкджэк c возможностью удваивать
    '''
    def __init__(self, natural=True, matrix_q=None, matrix_reward=None, use_matrix_reward=False):
        self._env = gym.make('Blackjack-v0', natural=natural)
        self._env1 = gym.make('Blackjack-v0', natural=natural)
#       Матрица полезности действий
        if isinstance(matrix_q, np.ndarray):
            size = matrix_q.shape
            assert size == (18, 3), 'Размер матрицы полезности должен быть 18x3'
            self._matrix_q = matrix_q
        else:
            self._matrix_q = np.random.rand(18,3)
#             self._matrix_q[:,2] = 10
#             self._matrix_q = np.ones((18,3))
#         Минимально возможное значнение карт на руках
        self._min_state = 4
        self._mean_reward = 0
        self._current_state = self._min_state
        
        self._use_matrix_reward = use_matrix_reward
        if self._use_matrix_reward:
            if isinstance(matrix_reward, np.ndarray):
                size = matrix_reward.shape
                assert size == (18, 3), 'Размер матрицы награды должен быть 18x3'
                self._matrix_reward = matrix_reward
            else:
                self._solve_matrix_reward()
        
    def game(self, n_games = 10000, matrix_q=None):
        '''Инициализации n_games партий'''
        if isinstance(matrix_q, np.ndarray):
             self.replace_matrix_q(matrix_q)
        sum_reward = 0
        for i in range(n_games):
            self.reset_game()
            done = False
            while done == False:
                action = self.choice_action(self._current_state)
                new_state, reward, done = self.get_new_state(action)
                if done == False:
                    self._current_state = new_state
            sum_reward += reward    
        self._mean_reward = sum_reward / n_games
    
    def reset_game(self,):
        """"""
        state = self._env.reset()
        self._current_state = state[0]
        
    def get_current_state(self,):
        return self._current_state
    
    def choice_action(self, current_state):
        '''
        Выбор действия по матрице полезности от текущего состояния
        current_state - текущее состояние
        '''
        row = self.convert_state_action_to_index_matrix_q(current_state, 0)[0]
        return np.argmax(self._matrix_q[row,:])
    
    def get_new_state(self, action):
        '''Возвращение нового состояния, награды и флага состояния игры после действия'''
        if action == 2:
            self._env.step(1)
            state, reward, done, _ = self._env.step(0)
            reward *= 2
                
        else:
            state, reward, done, _ = self._env.step(action)
        return state[0], reward, done
    
    def _get_reward(self, state, action, n_games=10000):
        sum_ = 0
        for i in range(n_games):
            self._env1.reset()
            self._env1.player = [state]
            if action == 2:
                self._env1.step(1)
                st, rew, done, _ = self._env1.step(0)
                rew *= 2
            else:
                st, rew, done, _ = self._env1.step(action)
            sum_ += rew
        return sum_/n_games
    
    def _solve_matrix_reward(self):
        print("Расчёт матрицы ожидания награды")
        self._matrix_reward = np.zeros_like(self._matrix_q)
        for i in tqdm(range(self._matrix_reward.shape[0])):
            for j in range(self._matrix_reward.shape[1]):
                self._matrix_reward[i,j] = self._get_reward(i + 4, j)
        
    def get_matrix_reward(self):
        return self._matrix_reward
    
    def get_new_state_from_lerning(self, current_state, action):
        '''Возвращение нового состояния, награды и флага состояния игры после действия'''
        if self._use_matrix_reward:
            if action == 2:
                self._env.step(1)
                state, reward, done, _ = self._env.step(0)
            else:
                state, reward, done, _ = self._env.step(action)
            reward = self._matrix_reward[self.convert_state_action_to_index_matrix_q(current_state, action)]
            return state[0], reward, done
        else:
            return self.get_new_state(action)
    
    def convert_state_action_to_index_matrix_q(self, state, action):
        """Конвертация состояния и действия в позицию строки и столбца матрицы полезности."""
        return state - self._min_state, action

    
class BlackjackModels():
    '''
    Модели игры в блекджек
        "simple" - модель простой игры
        "double" - модель простой игры с удвоением
    '''
    def __init__(self):
        pass
        
#         Начальные значения
     
    def get_matrix_q(self, name_model):
        return self._matrix_q[name_model]