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
    
    enum RotationState: Int {
        case Idle = 1
        case WaitingForFirstDegree = 2
        case WaitingForSecondDegree = 3
        case RotationInProgress = 4
    }
    
    // Core Motion variables.
    private let motionManager = CMMotionManager()
    private var motionQueue = OperationQueue()
    private static let motionFrequency = 1.0/30.0
    private let mpQueue = DispatchQueue(label: "Motion Processing Queue", qos: .userInitiated , attributes: [], autoreleaseFrequency: .inherit, target: nil)
    private var delegate: MotionObserverDelegate?
    
    private var currState: RotationState = .Idle
    private var rangeToRotate: Int = 0
    private var firstDegree: Int = 0
    private var secondDegree: Int = 0
    private var prevDegree: Int = 0
    private var direction: Int = 0
    private var wrongDirectionMode: Bool = false
    private var degreesRotated: Int = 0
    private static let degreeTolerance = 20
    
    init() {}
    
    // returns smallest the difference (a-b) in degrees between two angles that are close to each other taking into account roll over from 360 to 0.
    private static func smallestDegreeDiff(_ a: Int, _ b: Int) -> Int {
        if abs(a-b) > 360 - abs(a-b) {
            // Roll over.
            return a-b >= 0 ? abs(a-b) - 360 : 360 - abs(a-b)
        }
        return a - b
    }
    
    // differenceAlongDirection calculates the difference (a-b) in degrees between two angles along the given direction. Direction can be positive (i.e. 0->360 clockwise) or negative (i.e. 360->0 counterclockwise).
    private static func differenceAlongDirection(_ a: Int, _ b: Int, positive: Bool) -> Int {
        if positive {
            return a-b >= 0 ? a-b : 360 - (b-a)
        }
        return b-a >= 0 ? b-a: 360 - (a-b)
    }
    
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
            self.handleUserRotation(heading: Int(validData.heading))
        })
    }
        
    func stopMotionUpdates() {
        if !self.motionManager.isDeviceMotionActive {
            return
        }
        self.motionManager.stopDeviceMotionUpdates()
    }
    
    // ensureUserRotates ensures that user is rotating given range from current heading value.
    func ensureUserRotates(range: Int) {
        self.mpQueue.async {
            self.resetData()
            self.rangeToRotate = range
            self.currState = .WaitingForFirstDegree
        }
    }
    
    // handleUserRotation is a helper function to ensure user rotates from current position to
    // the given range successfully.
    private func handleUserRotation(heading: Int) {
        self.mpQueue.async {
            switch self.currState {
            case .Idle:
                break
            case .WaitingForFirstDegree:
                self.firstDegree = heading
                self.currState = .WaitingForSecondDegree
            case .WaitingForSecondDegree:
                if abs(MotionManager.smallestDegreeDiff(heading, self.firstDegree)) < MotionManager.degreeTolerance {
                    break
                }
                self.secondDegree = heading
                self.prevDegree = heading
                self.direction = MotionManager.smallestDegreeDiff(self.secondDegree, self.firstDegree)
                self.delegate?.userHasStartedRotating()
                self.currState = .RotationInProgress
            case .RotationInProgress:
                let prevDiff = MotionManager.smallestDegreeDiff(heading, self.prevDegree)
                if abs(prevDiff) < MotionManager.degreeTolerance {
                    // No need to update prevDegree.
                    break
                }
                
                // test that direction of user movement is correct.
                if self.direction * prevDiff >= 0 {
                    // User moving in right direction.
                    if self.wrongDirectionMode {
                        self.wrongDirectionMode = false
                        self.delegate?.userHasStartedRotating()
                    }
                    self.degreesRotated += MotionManager.differenceAlongDirection(heading, self.prevDegree, positive: self.direction > 0)
                    self.prevDegree = heading
                } else {
                    // User moving in wrong direction.
                    if !self.wrongDirectionMode {
                        self.wrongDirectionMode = true
                        self.delegate?.wrongRotationDirection()
                    }
                    self.degreesRotated -= MotionManager.differenceAlongDirection(heading, self.prevDegree, positive: self.direction < 0)
                    self.prevDegree = heading
                    
                    // Do not provide heading value.
                    return
                }
                
                // test if rotation complete.
                if self.degreesRotated >= self.rangeToRotate - MotionManager.degreeTolerance {
                    self.currState = .Idle
                    self.delegate?.rotationComplete()
                }
            }
            self.delegate?.updatedHeading(heading: heading)
        }
    }
    
    private func resetData() {
        self.rangeToRotate = 0
        self.firstDegree = 0
        self.secondDegree = 0
        self.prevDegree = 0
        self.direction = 0
        self.wrongDirectionMode = false
        self.degreesRotated = 0
    }
    
}
