//
//  SkinToneAnalyzer.swift
//  FaceMagik
//
//  Created by Addarsh Chandrasekar on 1/9/21.
//

import UIKit

class SkinToneAnalyzer: AssessFaceControllerDelegate {
    struct FaceValues {
        var heading: Int
        var leftCheekPercentValue: Int
        var rightCheekPercentValue: Int
    }
    
    private let headingQueue = DispatchQueue(label: "Heading Queue to analyze skin tone", qos: .userInitiated , attributes: [], autoreleaseFrequency: .inherit, target: nil)
    private var currLeftCheekPercentValue: Int?
    private var currRightCheekPercentValue: Int?
    private var headingValuesMap: [Int: FaceValues] = [:]
    
    init() {}
    
    func handleUpdatedHeading(heading: Int) {
        self.headingQueue.async {
            guard let leftCheekPercentValue = self.currLeftCheekPercentValue else {
                return
            }
            guard let rightCheekPercentValue = self.currRightCheekPercentValue else {
                return
            }
            if self.headingValuesMap[heading] != nil {
                return
            }
            self.headingValuesMap[heading] = FaceValues(heading: heading, leftCheekPercentValue: leftCheekPercentValue, rightCheekPercentValue: rightCheekPercentValue)
        }
    }
    
    func handleUpdatedImageValues(leftCheekPercentValue: Int, rightCheekPercentValue: Int) {
        self.headingQueue.async {
            self.currLeftCheekPercentValue = leftCheekPercentValue
            self.currRightCheekPercentValue = rightCheekPercentValue
        }
    }
    
    // estimatePrimaryLightDirection estimates primary light direction (degrees) from given heading and values data that span 360 degrees.
    func estimatePrimaryLightDirection() -> Int {
        self.headingQueue.sync {
            
            let values: [FaceValues] = self.headingValuesMap.map{$1}
            
            // Filter degrees with similar left and right cheek intensities.
            let filteredValues = values.filter { fv in abs(fv.leftCheekPercentValue - fv.rightCheekPercentValue) < 10
            }
            if filteredValues.count < 2 {
                return 0
            }
        
            var sum :Float = 0
            for v in values {
                sum += Float(v.leftCheekPercentValue)
                sum += Float(v.rightCheekPercentValue)
            }
            return Int(Float(sum)/Float(values.count*2))
        }
    }
}
