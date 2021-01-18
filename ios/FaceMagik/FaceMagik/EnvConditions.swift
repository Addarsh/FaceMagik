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
    private var delegate: EnvObserverDelegate?
    private var currTemp: Int = 0
    private var currISO: Int = 0
    private var currExposure: Int = 0
    private let envQueue = DispatchQueue(label: "Env Queue", qos: .userInitiated , attributes: [], autoreleaseFrequency: .inherit, target: nil)
    private var sensorMap: [Int: SensorValues] = [:]
    
    // updated Heading value of the user.
    func updatedHeading(heading: Int) {
        self.envQueue.async {
            if self.sensorMap[heading] != nil {
                return
            }
            self.sensorMap[heading] = SensorValues(iso: self.currISO, exposure: self.currExposure, temp: self.currTemp, sceneType: SceneType.Unknown)
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
    
    // observeLighting collects lighting parameters in surroundings.
    func observeLighting(delegate: EnvObserverDelegate?) {
        self.delegate = delegate
        self.envQueue.async {
            self.sensorMap = [:]
        }
    }
    
    // processLighting will process environment lighting using values in sensor map.
    func processLighting() {
        self.envQueue.async {
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
}
