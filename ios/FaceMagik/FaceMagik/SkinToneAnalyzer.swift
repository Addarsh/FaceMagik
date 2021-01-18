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
    
    // estimatePrimaryLightDirection estimates primary light direction (degrees) from given heading and values data.
    func estimatePrimaryLightDirection() -> Int {
        self.headingQueue.sync {
            
            let values: [FaceValues] = self.headingValuesMap.map{$1}
            
            // Filter degrees with similar left and right cheek intensities.
            let filteredValues = values.filter { fv in abs(fv.leftCheekPercentValue - fv.rightCheekPercentValue) < 10
            }
            if filteredValues.count < 2 {
                return 0
            }
            
            // Compute combined intensity cutoff.
            let intensityValues = filteredValues.sorted { fv1, fv2 in
                return fv1.leftCheekPercentValue + fv1.rightCheekPercentValue <= fv2.leftCheekPercentValue + fv2.rightCheekPercentValue
            }
            let intensityCutoff = intensityValues.last!.leftCheekPercentValue + intensityValues.last!.rightCheekPercentValue - 20
            
            // Filter valuess greater than given cutoff and sort in increasing order of intensity difference.
            let finalFilteredValues = intensityValues.filter { fv in
                return fv.leftCheekPercentValue + fv.rightCheekPercentValue >= intensityCutoff
            }
            let finalSortedValues = finalFilteredValues.sorted { fv1, fv2 in
                let diff1 = abs(fv1.leftCheekPercentValue - fv1.rightCheekPercentValue)
                let diff2 = abs(fv2.leftCheekPercentValue - fv2.rightCheekPercentValue)
                return diff1 <= diff2
            }
            
            return finalSortedValues[0].heading
        }
    }
}
