open System
open ValueIteration

// For more information see https://aka.ms/fsharp-console-apps
type State = int * int
type Action = 
    | Up = 0
    | Down = 1
    | Left = 2
    | Right = 3
type Reward = float
type BaseTransitionFunction = State->Action->((State * float) list)
type TransitionFunction = State->Action->State->float
type RewardFunction = State->Action->State->Reward
type MDP = { X : State list; A : Action list; p : TransitionFunction; r : RewardFunction }

let absorbtionMarker: State = (-1, -1)


let convertTransitionFunction (bt: BaseTransitionFunction) : TransitionFunction =
    let tf (s1: State) (a: Action) (s2: State) : float =
        let probs = bt s1 a
        let filtered = List.filter (fun x -> fst x = s2) probs
        match filtered with
        | n :: rest -> snd n
        | [] -> 0
    tf

let stateToString (s: State) : string =
    $"({fst s}, {snd s})"

let actionToString (a: Action) : string =
    match a with
    | Action.Up -> "Up"
    | Action.Down -> "Down"
    | Action.Left -> "Left"
    | Action.Right -> "Right"
    | _ -> ""

let rec getRandomElementWeighted (lst: 'a list) (weigths: float list) (r: float): 'a =
    if (r - (List.head weigths)) > 0 then
        getRandomElementWeighted (List.tail lst) (List.tail weigths) (r - List.head weigths)
    else
        List.head lst

let maximumX = 2
let maximumY = 2

let applyActionDirectly (stateFrom: State) (action: Action) : State =
    let keepWithinBorders (s: State) : State =
        max 0 (min maximumX (fst s)), max 0 (min maximumY (snd s))

    let step1 = match action with
                | Action.Up -> fst stateFrom, snd stateFrom + 1
                | Action.Down -> fst stateFrom, snd stateFrom - 1
                | Action.Left -> fst stateFrom - 1, snd stateFrom
                | Action.Right -> fst stateFrom + 1, snd stateFrom 
                | _ -> raise (ArgumentException("Error"))
        
    let step2 = keepWithinBorders step1
    step2

let turnRight (action: Action) : Action =
    match action with
    | Action.Up -> Action.Right
    | Action.Down -> Action.Left
    | Action.Left -> Action.Up
    | Action.Right -> Action.Down
    | _ -> raise (ArgumentException("Error"))

let turnLeft (action: Action) : Action =
    turnRight (turnRight (turnRight action))

let btf1 (stateFrom: State) (action: Action) : (State * float) list =
    if (stateFrom = absorbtionMarker) then
        [absorbtionMarker, 1]
    else if (stateFrom = (0,0) || stateFrom = (0,2)) then
        [absorbtionMarker, 1]
    else
        [(applyActionDirectly stateFrom action), 0.8 ; (applyActionDirectly stateFrom (turnRight action)), 0.1 ; (applyActionDirectly stateFrom (turnLeft action)), 0.1]

let tf1: TransitionFunction = convertTransitionFunction btf1
let rf1 sFrom a sTo: float =
    if (sFrom = (0,0) && sTo=absorbtionMarker) then
        10
    else if (sFrom = (0,2) && sTo=absorbtionMarker) then
        -10
    else if (sFrom = absorbtionMarker) then
        0
    else if (sFrom = sTo) then
        0
    else
        5

let si (s: State): int =
    if (s = absorbtionMarker) then
        9
    else
        (fst s)*3+(snd s)

let mdp_x = [(0,0); (0,1); (0,2); (1,0); (1,1); (1,2); (2,0); (2,1); (2,2) ; absorbtionMarker]
let mdp_a = [Action.Up ; Action.Down ; Action.Left ; Action.Right]

let mdp = {X = mdp_x ; A = mdp_a ; p = tf1 ; r = rf1 ;
si = si ; ai = int }

let Q = ValueIteration.valueIteration mdp 0.8 

for x in mdp.X do
    for a in mdp.A do
        printfn "%s %s: %.3f" (stateToString x) (actionToString a) Q[si x, int a]
