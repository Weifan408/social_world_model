import numpy as np



DIRECTION_MAP = {
    0: (0, -1),
    1: (0, 1),
    2: (1, 0),
    3: (-1, 0)
}



def decode_observation_old(obs, player_id):
    """
    00 = 'player_0_loc'
    01 = 'player_1_loc'
    02 = 'player_0_orientation_0'
    03 = 'player_0_orientation_1'
    04 = 'player_0_orientation_2'
    05 = 'player_0_orientation_3'
    06 = 'player_1_orientation_0'
    07 = 'player_1_orientation_1'
    08 = 'player_1_orientation_2'
    09 = 'player_1_orientation_3'
    # 10-14 static info, ignore 
    10= 'pot_loc'
    # 11 = 'counter_loc'
    # 12 = 'onion_disp_loc'
    # 13 = 'dish_disp_loc'
    # 14 = 'serve_loc'
    15 = 'onions_in_pot'
    16 = 'onions_cook_time'
    17 = 'onion_soup_loc'
    18 = 'dishes'
    19 = 'onions'
    20 = 'timestep' # ignore
    """
    from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import  OvercookedState, PlayerState, ObjectState


    class DummyOvercookedState(OvercookedState):
        def __init__(self, players, objects):
            self.players = tuple(players)
            self.objects = objects
    
    obs = np.transpose(obs, (2, 0, 1))  # Shape: (C, H, W)
    
    players_list = [None, None]
    players_pos = [[],[]]
    for pid in [0, 1]:
        loc_channel = 0 if pid == player_id else 1
        pos = np.unravel_index(np.argmax(obs[pid]), obs[pid].shape)
        pos = tuple(pos) if len(pos) > 0 else None
        players_pos[loc_channel] = pos

        orientation = None
        for d in range(4):
            idx = 2 + pid * 4 + d
            if np.round(obs[idx]).max() == 1:
                orientation = DIRECTION_MAP[d]
                break
        if orientation is None:
            orientation = DIRECTION_MAP[1]
        players_list[loc_channel] = PlayerState(pos, orientation)
    
    # --- Soup Features (normalized) ---
    soup_infos = {}
    objects = {}
    def decode_soup_info(idx, name, scale):
        discretized_obs = np.clip(np.round(obs[idx] * scale), 0, scale)
        positions = np.argwhere(discretized_obs > 0)
        if len(positions) > 0:
            for pos in positions:
                if tuple(pos) not in soup_infos:
                    soup_infos[tuple(pos)] = {}
                value = discretized_obs[tuple(pos)]
                soup_infos[tuple(pos)][name] = int(value)
    
    decode_soup_info(15, "onions_in_pot", 3) # 1/3 2/3 1  
    decode_soup_info(16, "onions_cook_time", 20)
    decode_soup_info(17, "onion_soup_loc", 1)
    
    if soup_infos:
        for pos in soup_infos:
            cook_time = soup_infos[pos].get("onions_cook_time", 0)
            onions_in_pot_num = soup_infos[pos].get("onions_in_pot", 0)
            onion_soup_loc = soup_infos[pos].get("onion_soup_loc", None)
            if onions_in_pot_num > 0:
                if onions_in_pot_num < 3:
                    cook_time = 0
                objects[pos] = ObjectState(
                    "soup", pos, state=('onion', onions_in_pot_num, cook_time)
                )
            elif onion_soup_loc is not None:
                objects[pos] = ObjectState("soup", pos, ['onion', 3, 20])
    
    # --- Object Locations ---
    object_features = ["dish", "onion"]
    for i, name in enumerate(object_features, start=18):
        discretized_obs = (obs[i] > 0.8).astype(int)
        positions = np.argwhere(discretized_obs == 1)
        if len(players_pos) > 0:
            for pos in positions:
                pos = tuple(pos)
                obj = ObjectState(name=name, position=pos)
                if pos in players_pos:
                    for id, p in enumerate(players_pos):
                        if pos == p:
                            if not players_list[id].has_object():
                                players_list[id].set_object(obj)
                            break
                else:
                    objects[pos] = obj

    return DummyOvercookedState(players_list, objects)

def decode_observation_new(obs, player_id):
    from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.overcooked_mdp import  OvercookedState, PlayerState, ObjectState,SoupState


    class DummyOvercookedState(OvercookedState):
        def __init__(self, players, objects):
            self.players = tuple(players)
            self.objects = objects
            
        
    class DummySoupState(SoupState):
        def __init__(self, position, ingredients, remaining_time=-1):
            super().__init__(position, ingredients)
            self.remaining_time = remaining_time
        
        @property
        def is_ready(self):
            if self.remaining_time < 0:
                return False
            return self.remaining_time == 0
    """
    00 = 'player_0_loc'
    01 = 'player_1_loc'
    02 = 'player_0_orientation_0'
    03 = 'player_0_orientation_1'
    04 = 'player_0_orientation_2'
    05 = 'player_0_orientation_3'
    06 = 'player_1_orientation_0'
    07 = 'player_1_orientation_1'
    08 = 'player_1_orientation_2'
    09 = 'player_1_orientation_3'
    # 10-15 static info, ignore 
    # 10= 'pot_loc'
    # 11 = 'counter_loc'
    # 12 = 'onion_disp_loc'
    # 13 = 'tomato_disp_loc'
    # 14 = 'dish_disp_loc'
    # 15 = 'serve_loc'
    16 = 'onions_in_pot'
    17 = 'tomatoes_in_pot'
    18 = 'onions_in_soup'
    19 = 'tomatoes_in_soup'
    20 = 'soup_cook_time_remaining'
    21 = 'dishes'
    22 = 'onions'
    23 = 'tomatoes'
    24 = 'urgency' # ignore
    25 = 'timestep' # ignore
    """
    obs = np.transpose(obs, (2, 0, 1))  # Shape: (C, H, W)
    
    players_list = [None, None]
    players_pos = [[],[]]
    for pid in [0, 1]:
        loc_channel = 0 if pid == player_id else 1
        pos = np.unravel_index(np.argmax(obs[pid]), obs[pid].shape)
        pos = tuple(pos) if len(pos) > 0 else None
        players_pos[loc_channel] = pos

        orientation = None
        for d in range(4):
            idx = 2 + pid * 4 + d
            if np.round(obs[idx]).max() == 1:
                orientation = DIRECTION_MAP[d]
                break
        if orientation is None:
            orientation = DIRECTION_MAP[1]
        players_list[loc_channel] = PlayerState(pos, orientation)

    # --- Soup Features (normalized) ---
    objects = {}
    soup_infos = {}
    def decode_soup_info(idx, name, scale):
        discretized_obs = np.clip(np.round(obs[idx] * scale), 0, 3)
        positions = np.argwhere(discretized_obs > 0)
        if len(positions) > 0:
            for pos in positions:
                if tuple(pos) not in soup_infos:
                    soup_infos[tuple(pos)] = {}
                value = discretized_obs[tuple(pos)]
                soup_infos[tuple(pos)][name] = int(value)
    
    decode_soup_info(16, "onions_in_pot", 3) # 1/3 2/3 1  
    decode_soup_info(17, "tomatoes_in_pot", 3)
    decode_soup_info(18, "onions_in_soup", 3)
    decode_soup_info(19, "tomatoes_in_soup", 3)
    
    if soup_infos:
        for pos in soup_infos:
            onions_in_soup_num = soup_infos[pos].get("onions_in_soup", 0)
            tomatoes_in_soup_num = soup_infos[pos].get("tomatoes_in_soup", 0)
            onions_in_pot_num = soup_infos[pos].get("onions_in_pot", 0)
            tomatoes_in_pot_num = soup_infos[pos].get("tomatoes_in_pot", 0)
            if onions_in_soup_num > 0 or tomatoes_in_soup_num > 0:
                remaining_time = obs[20][pos] if obs[20][pos]>1e-3 else 0
                _ingredients= [ObjectState('onion', pos)] * onions_in_soup_num + [ObjectState('tomato', pos)] * tomatoes_in_soup_num
                obj = DummySoupState(
                    pos, ingredients=_ingredients[:3],
                    remaining_time=remaining_time
                )
                if pos in players_pos:
                    for id, p in enumerate(players_pos):
                        if pos == p:
                            players_list[id].set_object(obj)
                            break
                else:
                    objects[pos] = obj
            else:
                remaining_time = -1
                _ingredients=[ObjectState('onion', pos)] * onions_in_pot_num + [ObjectState('tomato', pos)] * tomatoes_in_pot_num
                objects[pos] = DummySoupState(
                    pos, ingredients=_ingredients[:3],
                    remaining_time=remaining_time
                )
    
    # --- Object Locations ---
    object_features = ["dish", "onion", "tomato"]
    for i, name in enumerate(object_features, start=21):
        discretized_obs = (obs[i] > 0.8).astype(int)
        positions = np.argwhere(discretized_obs == 1)
        if len(players_pos) > 0:
            for pos in positions:
                pos = tuple(pos)
                obj = ObjectState(name=name, position=pos)
                if pos in players_pos:
                    for id, p in enumerate(players_pos):
                        if pos == p:
                            if not players_list[id].has_object():
                                players_list[id].set_object(obj)
                            break
                else:
                    objects[pos] = obj
                # if pos in players_pos:
                #     for id, p in enumerate(players_pos):
                #         if pos == p:
                #             players_list[id].set_object(obj)
                #             break

    return DummyOvercookedState(players_list, objects)