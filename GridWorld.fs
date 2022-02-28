module GridWorld

open System
open ValueIteration
open QLearning
open RLCore

type State = int * int
type Action = 
    | Up = 0
    | Down = 1
    | Left = 2
    | Right = 3
type Reward = float

// The state we go to after having been absorbed. Nothing can be done here.
let absorbtionMarker: State = (-1, -1)

let stateToString (s: State) : string =
    $"({fst s}, {snd s})"

let actionToString (a: Action) : string =
    match a with
    | Action.Up -> "Up"
    | Action.Down -> "Down"
    | Action.Left -> "Left"
    | Action.Right -> "Right"
    | _ -> ""

/// Gets all the states for a world of the specified size
/// It is simply all combinations of x and y as well as the absorbtionMarker
let getStates xSize ySize (blockedStates: State list) =
    absorbtionMarker :: // absorbtionMarker must me included
    [for x in 0 .. xSize-1 do
        for y in 0 .. ySize-1 -> (x, y) ]

let getActions: Action list = [Action.Up; Action.Down; Action.Left; Action.Right]

let turnRight action : Action =
    match action with
    | Action.Up -> Action.Right
    | Action.Down -> Action.Left
    | Action.Left -> Action.Up
    | Action.Right -> Action.Down
    | _ -> raise (ArgumentException("Error"))

let turnLeft action : Action =
    action |> turnRight |> turnRight |> turnRight

let keepWithinBorders xSize ySize (s: State) : State =
    max 0 (min (xSize-1) (fst s)), max 0 (min (ySize-1) (snd s))

/// Trying to move to or from a blocked state is not possible and then nothing is done
let disallowBlockedState (blockedStates: State list) stateFrom stateTo : State =
    if (List.contains stateTo blockedStates || List.contains stateFrom blockedStates) then
        stateFrom else stateTo

let applyActionSimple action stateFrom : State =
    match action with
    | Action.Up -> fst stateFrom, snd stateFrom + 1
    | Action.Down -> fst stateFrom, snd stateFrom - 1
    | Action.Left -> fst stateFrom - 1, snd stateFrom
    | Action.Right -> fst stateFrom + 1, snd stateFrom 
    | _ -> raise (ArgumentException("Error"))

let applyActionDirectly xSize ySize (blockedStates: State list) stateFrom action : State =
    stateFrom |>
    applyActionSimple action |>
    keepWithinBorders xSize ySize |>
    disallowBlockedState blockedStates stateFrom

let baseTransitionFunction xSize ySize (blockedStates: State list) (absorbitionStates: State list) (turnProbability: float) stateFrom action : (State * float) list =
    let apply = applyActionDirectly xSize ySize blockedStates
    if (stateFrom = absorbtionMarker) then // You cannot move away from the absorbtionMarker
        [absorbtionMarker, 1] 
    else if (List.contains stateFrom absorbitionStates) then // If this is an absorbtion state then we go to the absorbtion marker
        [absorbtionMarker, 1] 
    else
        [(apply stateFrom action), 1.0 - (2.0 * turnProbability) ; // Doing the action
         (apply stateFrom (turnRight action)), turnProbability ; // Turning right
          (apply stateFrom (turnLeft action)), turnProbability] // Turning left

let transitionfunction xSize ySize (blockedStates: State list) (absorbitionStates: Map<State, Reward>) (turnProbability: float) : State->Action->State->float =
    let abStates = absorbitionStates |> Map.toList |> List.map fst
    let bt = baseTransitionFunction xSize ySize blockedStates abStates turnProbability
    RLCore.convertTransitionFunction bt

let transitionfunctionStochastic xSize ySize (blockedStates: State list) (absorbitionStates: Map<State, Reward>) (turnProbability: float) : State->Action->State =
    let abStates = absorbitionStates |> Map.toList |> List.map fst
    let bt = baseTransitionFunction xSize ySize blockedStates abStates turnProbability
    QLearning.convertTransitionFunctionStochastic bt

let rewardFunction (absorbitionStates: Map<State, Reward>) (livingReward: float) stateFrom action stateTo : Reward =
    if (stateFrom = absorbtionMarker) then
        0
    else
        match Map.tryFind stateFrom absorbitionStates with
        | Some(reward) -> reward
        | None -> livingReward

let stateIndex xSize ySize state : int =
    if (state = absorbtionMarker) then
        xSize*ySize
    else
        (fst state) * (ySize) + (snd state)

let actionIndex (action: Action) : int = int action

let printQ maxX maxY (Q: float[,]) : unit =
    for y in (maxY-1) .. -1 .. 0 do
        let vals: string = [for x in 0..(maxX-1) do getStateValue (stateIndex maxX maxY) (x,y) Q] |>
                                List.map (sprintf "+%.2f") |>
                                String.concat " " |>
                                fun s -> s.Replace("+-", "-")
        printfn "%s" vals
    0.0 |> ignore