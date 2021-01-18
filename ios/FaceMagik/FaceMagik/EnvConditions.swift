//
//  EnvConditions.swift
//  FaceMagik
//
//  Created by Addarsh Chandrasekar on 1/3/21.
//

import Photos
import CoreMotion

enum SceneType: Int, Codable {
    case Indoors = 1
    case Outdoors = 2
    case Unknown = 3
}

struct SensorValues {
    var iso: Int
    var exposure: Int
    var temp: Int
    var sceneType: SceneType
}

class EnvConditions: NSObject, EnvObserver {
    enum State: Int {
        case Idle = 1
        case WaitingForFirstDegree = 2
        case WaitingForSecondDegree = 3
        case CollectionInProgress = 4
    }
    
    private var delegate: EnvObserverDelegate?
    private var currTemp: Int = 0
    private var currISO: Int = 0
    private var currExposure: Int = 0
    private var currState: State = .Idle
    private let envQueue = DispatchQueue(label: "Env Queue", qos: .userInitiated , attributes: [], autoreleaseFrequency: .inherit, target: nil)
    private var sensorMap: [Int: SensorValues] = [:]
    
    // collection variables.
    private var rangeToRotate: Int = 0
    private var firstDegree: Int = 0
    private var secondDegree: Int = 0
    private var prevDegree: Int = 0
    private var direction: Int = 0
    private var wrongDirectionMode: Bool = false
    private var degreesRotated: Int = 0
    private static let degreeTolerance = 20

    
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
    
    // updated Heading value of the user.
    func updatedHeading(heading: Int) {
        self.envQueue.async {
            self.handleUserRotation(heading: heading)
        }
    }
    
    func updatedISO(iso: Int) {
        self.envQueue.async {
            self.currISO = iso
        }
    }
    
    func updatedExposure(exposure: Int) {
        self.envQueue.async {
            self.currExposure = exposure
        }
    }
    
    func updatedColorTemp(temp: Int) {
        self.envQueue.async {
            self.currTemp = temp
        }
    }
    
    // handleUserRotation is a helper function handle state associated with user rotation.
    private func handleUserRotation(heading: Int) {
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
        if self.sensorMap[heading] != nil {
            return
        }
        
        self.sensorMap[heading] = SensorValues(iso: self.currISO, exposure: self.currExposure, temp: self.currTemp, sceneType: SceneType.Unknown)
        
        if self.degreesRotated >= self.rangeToRotate - EnvConditions.degreeTolerance {
            self.currState = .Idle
            
            self.delegate?.motionUpdateComplete()
            self.processLighting()
        }
    }
    
    // observeUserRotation manages user rotation and collects lighting parameters in surroundings.
    func observerUserRotation(range: Int, delegate: EnvObserverDelegate?) {
        self.delegate = delegate
        self.resetData()
        self.envQueue.async {
            self.rangeToRotate = range
            self.currState = .WaitingForFirstDegree
        }
    }
    
    func stopObserving() {
        self.envQueue.async {
            self.resetData()
            self.currState = .Idle
        }
    }
    
    private func resetData() {
        self.sensorMap = [:]
        self.rangeToRotate = 0
        self.firstDegree = 0
        self.secondDegree = 0
        self.prevDegree = 0
        self.direction = 0
        self.wrongDirectionMode = false
        self.degreesRotated = 0
    }
    
    // processLighting will process environment lighting using values in sensor map.
    private func processLighting() {
        var avgTemp: Float = 0
        var avgISO: Float = 0
        var avgExposure: Float = 0
        for (_, readouts) in self.sensorMap {
            avgTemp += Float(readouts.temp)
            avgISO += Float(readouts.iso)
            avgExposure += Float(readouts.exposure)
        }
        
        let kCount = Float(self.sensorMap.count)
        avgTemp /= kCount
        avgISO /= kCount
        avgExposure /= kCount
        
        if avgTemp < 4000 {
            self.delegate?.badColorTemperature()
            return
        }
        if avgExposure >= 45 {
            self.delegate?.possiblyOutdoors()
            return
        }
        if avgISO < 200 {
            self.delegate?.tooBright()
        }
    }
}
