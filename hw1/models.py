import gym
import numpy as np
import multiprocessing as mp
from time import time
from tqdm.notebook import tqdm
from gym.envs.toy_text import BlackjackEnv
import os
import random

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
        self.shufle()
        
        self._model_score = model_score
        self._model_score_cards = {"plus_minus":np.array([-1, 1, 1, 1, 1, 1, 0, 0, 0, -1]),
                                   "half":np.array([-1, 0.5, 1, 1, 1.5, 1, 0.5, 0, -0.5, -1])}
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

class ModelUtils:
    '''
    Методы для моделей игр в блэкджэк
    '''   
    def get_mean_reward(self,):
        """Возвращение среднего выигрыша"""
        return self._mean_reward   
    
    def get_matrix_q(self,):
        """Возвращение текущей матрицы полезности"""
        return self._matrix_q
    
    def get_current_state(self,):
        return self._current_state
    
    def get_matrix_actions(self, matrix_q, num_col=1):
        """Вывод матрицы действий в зависимости от состояний"""
        return np.argmax(matrix_q, axis=1).reshape(-1, num_col)
    
    def convert_state_action_to_index_matrix_q(self, state, action):
        """
        Конвертация состояния и действия в позицию строки и столбца матрицы полезности.
        """
        return state, action
    
    def choice_action(self, current_state):
        '''
        Выбор действия по матрице полезности от текущего состояния
        current_state - текущее состояние
        '''
        row, col = self.convert_state_action_to_index_matrix_q(current_state, 0)
        return np.argmax(self._matrix_q[row,:])
    
    
    def step(self, action):
        assert self.action_space.contains(action)
        if action:  # hit: add a card to players hand and return
            self.player.append(self.draw_card(self.deck))
            if self.is_bust(self.player):
                done = True
                reward = -1.
            else:
                done = False
                reward = 0.
        else:  # stick: play out the dealers hand, and score
            done = True
            while self.sum_hand(self.dealer) < 17:
                self.dealer.append(self.draw_card(self.deck))
            reward = self.cmp(self.score(self.player), self.score(self.dealer))
            if self.natural and self.is_natural(self.player) and reward == 1.:
                reward = 1.5
        return self._get_obs(), reward, done, {}
    
    def reset(self):
        if self.deck.need_shufle():
            self.deck.shufle()
        self.dealer = self.draw_hand(self.deck)
        self.player = self.draw_hand(self.deck)
        return self._get_obs()
    
    
    def draw_card(self, deck):
        '''deck - объект класса колоды карт'''
        return int(deck.get_card_from_deck(1))
    
    def draw_hand(self, deck):
        '''deck - объект класса колоды карт'''
        return list(deck.get_card_from_deck(2))
    
    '''
    Ниже приведены функции взятые из доп.функций класса блэкджек
    '''
    
    def cmp(self, a, b):
        return float(a > b) - float(a < b)

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

class SimpleModel(ModelUtils, BlackjackEnv):
    '''
    Простая модель игры в блэкджэк
    '''
    def __init__(self, natural=True, matrix_q=None, multiproc_game=False, n_cpu=mp.cpu_count()):
        self.deck = Deck(shufle_treshold=52)
        super(SimpleModel, self).__init__(natural)
        self.space_action = np.arange(2)
#         Минимально возможное значнение карт на руках
        self._min_state = 4
        self._mean_reward = 0
        self._current_state = self._min_state

        self.replace_matrix_q(matrix_q)
        
        self._mulitiproc_game = multiproc_game
        self._n_cpu=n_cpu

    def _one_game_until_reset_deck(self,):
        sum_reward = 0
        party = 0
        self.reset_game()
        while self.deck.need_shufle() == False:
            done = False
            while done == False:
                action = self.choice_action(self._current_state)
                new_state, reward, done = self.get_new_state(action)
                if done == False:
                    self._current_state = new_state
            sum_reward += reward
            party += 1
            self.reset_game()
        return sum_reward, party  
    
    def _game_multiproc(self, n_games=10000, seed=time()):
        count = 0
        reward = 0
        _seed = mp.current_process().name.split('-')
#         print(_seed)
        if _seed[0] == 'MainProcess':
            _seed = seed
        else:
            _seed = _seed[1]
        np.random.seed(int(_seed))
        while count < n_games:
            rew, par = self._one_game_until_reset_deck()
            reward += rew
            count += par 
#         print(mp.current_process().name)
        return reward, count
    
    def _game_one_proc(self, n_games = 10000):
        '''Инициализации n_games партий'''
        sum_reward = 0
        for i in range(n_games):
            self.reset_game()
            done = False
            while not done:
                action = self.choice_action(self._current_state)
                new_state, reward, done = self.get_new_state(action)
                if not done:
                    self._current_state = new_state
            sum_reward += reward    
        self._mean_reward = sum_reward / n_games
    
    def game(self, n_games = 10000, matrix_q=None):
        '''Инициализации n_games партий'''
        if isinstance(matrix_q, np.ndarray):
             self.replace_matrix_q(matrix_q)
        if self._mulitiproc_game:
            pool = mp.Pool(processes=self._n_cpu)
            chunk = int(n_games/self._n_cpu)
            result = pool.map(self._game_multiproc, np.ones(self._n_cpu)*chunk)
            _sum = np.sum(result, axis=0)
            self._mean_reward = _sum[0]/_sum[1]
        else:
            self._game_one_proc(n_games=n_games)

    def reset_game(self,):
        """"""
        state = self.reset()
        self._current_state = state[0]
    
    def get_new_state(self, action):
        '''Возвращение нового состояния, награды и флага состояния игры после действия'''
        state, reward, done, _ = self.step(action)
        return state[0], reward, done
    
    def convert_state_action_to_index_matrix_q(self, state, action):
        """Конвертация состояния и действия в позицию строки и столбца матрицы полезности."""
        return (self.dealer[0] - 1) * 18 + state - self._min_state, action
    
    def replace_matrix_q(self, new_matrix_q):
        """
        Замена текущей матрицы полезности
            new_matrix_q - матрица для замены
        """
        if isinstance(new_matrix_q, np.ndarray):
            size = new_matrix_q.shape
            assert size == (180, 2), 'Размер матрицы полезности должен быть 180x2'
            self._matrix_q = new_matrix_q
        else:
            self._matrix_q = np.random.rand(180,2)
        

class DoubleModel(SimpleModel):
    '''
    Простая модель игры в блэкджэк c возможнотью удвоения (action = 2)
    '''
    def __init__(self, natural=True, matrix_q=None, multiproc_game=False, n_cpu=mp.cpu_count()):
        super(DoubleModel, self).__init__(natural, matrix_q, multiproc_game, n_cpu)
        self.space_action = np.arange(3)
    
    def get_new_state(self, action):
        '''Возвращение нового состояния, награды и флага состояния игры после действия'''
        if action == 2:
            self.step(1)
            state, reward, done, _ = self.step(0)
            reward *= 2
                
        else:
            state, reward, done, _ = self.step(action)
        return state[0], reward, done
    
    def replace_matrix_q(self, new_matrix_q):
        """
        Замена текущей матрицы полезности
            new_matrix_q - матрица для замены
        """
        if isinstance(new_matrix_q, np.ndarray):
            size = new_matrix_q.shape
            assert size == (180, 3), 'Размер матрицы полезности должен быть 18x3'
            self._matrix_q = new_matrix_q
        else:
            self._matrix_q = np.random.rand(180,3)        
        

class СountingModel(DoubleModel):
    '''
    Модель игры в блэкджэк с подсчетом карт
    '''
    def __init__(self, natural=True, matrix_q=None, multiproc_game=False, n_cpu=mp.cpu_count(),
                 shufle_treshold=15, counts_shuffle=1, model_score="plus_minus"):
        
        self.deck = Deck(shufle_treshold=shufle_treshold,
                         counts_shuffle=counts_shuffle,
                         model_score=model_score)

        super(СountingModel, self).__init__(natural, matrix_q, multiproc_game, n_cpu)
    
    def replace_matrix_q(self, new_matrix_q):
        """
        Замена текущей матрицы полезности
            new_matrix_q - матрица для замены
        """
        if isinstance(new_matrix_q, np.ndarray):
            size = new_matrix_q.shape
            assert size == (738, 3), 'Размер матрицы полезности должен быть 738x3'
            self._matrix_q = new_matrix_q
        else:
            self._matrix_q = np.random.rand(738, 3)
    
    def convert_state_action_to_index_matrix_q(self, state, action):
        row = 18 * (self.deck.get_score_deck() + 20) + state - self._min_state
        return row, action      
 