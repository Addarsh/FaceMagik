//
//  MotionManager.swift
//  FaceMagik
//
//  Created by Addarsh Chandrasekar on 1/17/21.
//

import CoreMotion

class MotionManager: MotionObserver {
    enum Direction: Int {
        case Clockwise = 1
        case CounterClockwise = 2
        case Either = 3
    }
    
    // Core Motion variables.
    private let motionManager = CMMotionManager()
    private var motionQueue = OperationQueue()
    private static let motionFrequency = 1.0/30.0
    private var delegate: MotionObserverDelegate?
    
    private var rangeToRotate: Int = 0
    private var firstDegree: Int = 0
    private var secondDegree: Int = 0
    private var prevDegree: Int = 0
    private var direction: Int = 0
    private var wrongDirectionMode: Bool = false
    private var degreesRotated: Int = 0
    private static let degreeTolerance = 20
    
    init() {}
    
    func startMotionUpdates(delegate: MotionObserverDelegate?) {
        if !self.motionManager.isDeviceMotionAvailable {
            print ("Device motion unavaible! Error!")
            return
        }
        if self.motionManager.isDeviceMotionActive {
            return
        }
        self.delegate = delegate
        self.motionManager.deviceMotionUpdateInterval = MotionManager.motionFrequency
        self.motionManager.startDeviceMotionUpdates(using: .xMagneticNorthZVertical, to: self.motionQueue, withHandler: { (data, error) in
            guard let validData = data else {
                return
            }
            self.delegate?.updatedHeading(heading: Int(validData.heading))
        })
    }
        
    func stopMotionUpdates() {
        if !self.motionManager.isDeviceMotionActive {
            return
        }
        self.motionManager.stopDeviceMotionUpdates()
    }
    
    // ensureUserRotates ensures that user is rotating given range from current heading value
    // in the given direction.
    private func ensureUserRotates(range: Int, dir: Direction) {
        
    }
    
    /*private func handleUserRotation(heading: Int) {
        switch self.currState {
        case .Idle:
            return
        case .WaitingForFirstDegree:
            self.firstDegree = heading
            self.currState = .WaitingForSecondDegree
        case .WaitingForSecondDegree:
            if abs(EnvConditions.smallestDegreeDiff(heading, self.firstDegree)) < EnvConditions.degreeTolerance {
                break
            }
            self.secondDegree = heading
            self.prevDegree = heading
            self.direction = EnvConditions.smallestDegreeDiff(self.secondDegree, self.firstDegree)
            self.delegate?.motionUpdating()
            self.currState = .CollectionInProgress
        case .CollectionInProgress:
            let prevDiff = EnvConditions.smallestDegreeDiff(heading, self.prevDegree)
            if abs(prevDiff) < EnvConditions.degreeTolerance {
                // No need to update prevDegree.
                break
            }
            
            // test that direction of user movement is correct.
            if self.direction * prevDiff >= 0 {
                // User moving in right direction.
                if self.wrongDirectionMode {
                    self.wrongDirectionMode = false
                    self.delegate?.motionUpdating()
                }
                self.degreesRotated += EnvConditions.differenceAlongDirection(heading, self.prevDegree, positive: self.direction > 0)
                self.prevDegree = heading
            } else {
                // User moving in wrong direction.
                if !self.wrongDirectionMode {
                    self.wrongDirectionMode = true
                    self.delegate?.wrongMotionDirection()
                }
                self.degreesRotated -= EnvConditions.differenceAlongDirection(heading, self.prevDegree, positive: self.direction < 0)
                self.prevDegree = heading
                return
            }
        }
    }*/
    
}
