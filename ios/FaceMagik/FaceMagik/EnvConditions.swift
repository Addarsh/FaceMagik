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
        case WaitingForFirstDegree = 1
        case WaitingForSecondDegree = 2
        case CollectionInProgress = 3
        case CollectionComplete = 4
    }
    
    private var delegate: EnvObserverDelegate?
    @objc private var cameraDevice: AVCaptureDevice!
    private var exposureObservation: NSKeyValueObservation?
    private var isoObservation: NSKeyValueObservation?
    private var tempObservation: NSKeyValueObservation?
    private var currTemp: Int = 0
    private var currISO: Int = 0
    private var currExposure: Int = 0
    private let envQueue = DispatchQueue(label: "Env Queue", qos: .userInitiated , attributes: [], autoreleaseFrequency: .inherit, target: nil)
    
    // Core Motion variables.
    private let motionManager = CMMotionManager()
    private var motionQueue = OperationQueue()
    private static let motionFrequency = 1.0/30.0
    private var sensorMap: [Int: SensorValues] = [:]
    private var currState: State = .WaitingForFirstDegree
    
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
    
    func observeLighting(device: AVCaptureDevice?, vc: EnvObserverDelegate?) {
        guard let dev = device else {
            return
        }
        self.cameraDevice = dev
        self.delegate = vc
        
        // Set initial values.
        self.envQueue.async {
            self.currExposure = Int(1/self.cameraDevice.exposureDuration.seconds)
            self.currISO = Int(self.cameraDevice.iso)
            self.currTemp = Int(self.cameraDevice.temperatureAndTintValues(for: self.cameraDevice.deviceWhiteBalanceGains).temperature)
        }
        self.delegate?.notifyExposureUpdate(newExpsosure: Int(1/self.cameraDevice.exposureDuration.seconds))
        self.delegate?.notifyISOUpdate(newISO: Int(self.cameraDevice.iso))
        self.delegate?.notifyTempUpdate(newTemp: Int(self.cameraDevice.temperatureAndTintValues(for: self.cameraDevice.deviceWhiteBalanceGains).temperature))
        
        // Start observing camera device exposureDuration.
        self.exposureObservation = observe(\.self.cameraDevice.exposureDuration, options: .new){
            object, change in
            guard let newVal = change.newValue else {
                return
            }
            self.envQueue.async {
                self.currExposure = Int(1/(newVal.seconds))
            }
            self.delegate?.notifyExposureUpdate(newExpsosure: Int(1/(newVal.seconds)))
        }
        
        // Start observing camera device white balance gains.
        self.isoObservation = observe(\.self.cameraDevice.iso, options: .new){
            obj, change in
            guard let newVal = change.newValue else {
                return
            }
            self.envQueue.async {
                self.currISO = Int(newVal)
            }
            self.delegate?.notifyISOUpdate(newISO: Int(newVal))
        }
        
        // Start observing camera device white balance gains.
        self.tempObservation = observe(\.self.cameraDevice.deviceWhiteBalanceGains, options: .new){
            obj, chng in
            let temp = self.cameraDevice.temperatureAndTintValues(for: self.cameraDevice.deviceWhiteBalanceGains).temperature
            self.envQueue.async {
                self.currTemp = Int(temp)
            }
            self.delegate?.notifyTempUpdate(newTemp: Int(temp))
        }
    }
    
    func startMotionUpdates(range: Int) {
        if !self.motionManager.isDeviceMotionAvailable {
            print ("Device motion unavaible! Error!")
            return
        }
        if self.motionManager.isDeviceMotionActive {
            return
        }
        
        self.currState = .WaitingForFirstDegree
        var firstDegree: Int = 0
        var secondDegree: Int = 0
        var prevDegree: Int = 0
        var direction: Int = 0
        var wrongDirectionMode: Bool = false
        var degreesRotated: Int = 0
        let degreeDelta = 20
        self.motionManager.deviceMotionUpdateInterval = EnvConditions.motionFrequency
        self.motionManager.startDeviceMotionUpdates(using: .xMagneticNorthZVertical, to: self.motionQueue, withHandler: { (data, error) in
            guard let validData = data else {
                return
            }
            let heading = Int(validData.heading)
            self.envQueue.async {
                switch self.currState {
                case .WaitingForFirstDegree:
                    firstDegree = heading
                    self.currState = .WaitingForSecondDegree
                case .WaitingForSecondDegree:
                    if abs(EnvConditions.smallestDegreeDiff(heading, firstDegree)) < degreeDelta {
                        break
                    }
                    secondDegree = heading
                    prevDegree = heading
                    direction = EnvConditions.smallestDegreeDiff(secondDegree, firstDegree)
                    self.delegate?.motionUpdating()
                    self.currState = .CollectionInProgress
                case .CollectionInProgress:
                    let prevDiff = EnvConditions.smallestDegreeDiff(heading, prevDegree)
                    if abs(prevDiff) < degreeDelta {
                        // No need to update prevDegree.
                        break
                    }
                    
                    // test that direction of user movement is correct.
                    if direction * prevDiff >= 0 {
                        // User moving in right direction.
                        if wrongDirectionMode {
                            wrongDirectionMode = false
                            self.delegate?.motionUpdating()
                        }
                        degreesRotated += EnvConditions.differenceAlongDirection(heading, prevDegree, positive: direction > 0)
                        prevDegree = heading
                    } else {
                        // User moving in wrong direction.
                        if !wrongDirectionMode {
                            wrongDirectionMode = true
                            self.delegate?.wrongMotionDirection()
                        }
                        degreesRotated -= EnvConditions.differenceAlongDirection(heading, prevDegree, positive: direction < 0)
                        prevDegree = heading
                        return
                    }
                case .CollectionComplete:
                    return
                }
                if self.sensorMap[heading] != nil {
                    return
                }
                self.delegate?.notifyHeading(heading: heading)
                
                self.sensorMap[heading] = SensorValues(iso: self.currISO, exposure: self.currExposure, temp: self.currTemp, sceneType: SceneType.Unknown)
                
                if degreesRotated >= range - degreeDelta {
                    self.currState = .CollectionComplete
                    
                    self.delegate?.motionUpdateComplete()
                    self.processLighting()
                }
            }
        })
    }
    
    func stopMotionUpdates() {
        if !self.motionManager.isDeviceMotionActive {
            return
        }
        self.motionManager.stopDeviceMotionUpdates()
        self.envQueue.async {
            self.sensorMap = [:]
        }
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
