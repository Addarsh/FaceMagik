//
//  EnvConditions.swift
//  FaceMagik
//
//  Created by Addarsh Chandrasekar on 1/3/21.
//

import Photos
import CoreMotion

class EnvConditions: NSObject, EnvObserver {    
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
    
    // returns smallest absolute difference (a-b) in degrees taking into account roll over from 360 to 0.
    private static func smallestDegreeDiff(_ a: Int, _ b: Int) -> Int {
        return abs(a-b) < 360 - abs(a-b) ? abs(a-b)  : 360 - abs(a-b)
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
        
        var firstDegree: Int? = nil
        var secondDegree: Int? = nil
        var lastDegree: Int? = nil
        var collectionComplete: Bool = false
        self.motionManager.deviceMotionUpdateInterval = EnvConditions.motionFrequency
        self.motionManager.startDeviceMotionUpdates(using: .xMagneticNorthZVertical, to: self.motionQueue, withHandler: { (data, error) in
            guard let validData = data else {
                return
            }
            let heading = Int(validData.heading)
            self.envQueue.async {
                if collectionComplete {
                    // Completed sensor data collection.
                    return
                }
                if firstDegree == nil {
                    firstDegree = heading
                    lastDegree = heading + 180 < 360 ? heading + 180 : heading - 180
                }
                if secondDegree == nil && EnvConditions.smallestDegreeDiff(heading, firstDegree!) >= 5 {
                    secondDegree = heading
                    self.delegate?.motionUpdating()
                }
                if self.sensorMap[heading] != nil {
                    return
                }
                self.sensorMap[heading] = SensorValues(iso: self.currISO, exposure: self.currExposure, temp: self.currTemp, sceneType: SceneType.Unknown)
                
                if abs(lastDegree! - heading) <= 5 {
                    collectionComplete = true
                    
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
        }
    }
}
