module QLearning

type StateIdx<'s> = 's->int
type ActionIdx<'a> = 'a->int

type Reward = float
type TransitionFunction<'s, 'a> = 's->'a->'s->float
type RewardFunction<'s, 'a> = 's->'a->'s->Reward
type MDP<'s, 'a> = { X : 's list; A : 'a list; p : TransitionFunction<'s, 'a>; r : RewardFunction<'s, 'a> ; si: StateIdx<'s> ; ai: ActionIdx<'a> }
