//
//  StateManager.swift
//  FaceMagik
//
//  Created by Addarsh Chandrasekar on 1/10/21.
//

import UIKit

class StateManager {
    enum State {
        case Unknown
        case StartTurnAround
        case TurnAroundComplete
        case EnvIsGood
    }
    private var currState: State = .Unknown
    private let queue = DispatchQueue(label: "State Queue", qos: .default, attributes: [], autoreleaseFrequency: .inherit, target: nil)
    
    init() {}
    
    func getState() -> State {
        var state: State = .Unknown
        self.queue.sync {
            state = self.currState
        }
        return state
    }
    
    func updateState(state: State) {
        self.queue.async {
            self.currState = state
        }
    }
    
}
