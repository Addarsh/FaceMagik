//
//  SkinToneAnalyzer.swift
//  FaceMagik
//
//  Created by Addarsh Chandrasekar on 1/9/21.
//

import UIKit

class SkinToneAnalyzer: AssessFaceControllerDelegate {
    struct FaceValues {
        var leftCheekPercentValue: Int
        var rightCheekPercentValue: Int
    }
    
    private let headingQueue = DispatchQueue(label: "Heading Queue to analyze skin tone", qos: .userInitiated , attributes: [], autoreleaseFrequency: .inherit, target: nil)
    private var currFaceValues: FaceValues?
    private var headingValuesMap: [Int:FaceValues] = [:]
    private var firstHeading: Int?
    
    init() {}
    
    func handleUpdatedHeading(heading: Int) {
        self.headingQueue.async {
            guard let faceValues = self.currFaceValues else {
                return
            }
            if self.firstHeading == nil {
                self.firstHeading = heading
            }
            if self.headingValuesMap[heading] != nil {
                return
            }
            self.headingValuesMap[heading] = faceValues
        }
    }
    
    func handleUpdatedImageValues(leftCheekPercentValue: Int, rightCheekPercentValue: Int) {
        self.headingQueue.async {
            self.currFaceValues = FaceValues(leftCheekPercentValue: leftCheekPercentValue, rightCheekPercentValue: rightCheekPercentValue)
        }
    }
    
    // estimatePrimaryLightDirection estimates primary light direction (degrees) from given heading and values data.
    func estimatePrimaryLightDirection() -> Int? {
        guard let firstHeading = self.firstHeading else {
            print ("first heading value missing")
            return nil
        }
        
        var count: Int = 0
        var degrees: [Int] = []
        while count < self.headingValuesMap.count {
            if self.headingValuesMap[firstHeading + count] == nil {
                continue
            }
            guard let lCheekPercentVal = self.headingValuesMap[firstHeading + count]?.leftCheekPercentValue else {
                continue
            }
            guard let rCheekPercentVal = self.headingValuesMap[firstHeading + count]?.rightCheekPercentValue else {
                continue
            }
            
            if abs(lCheekPercentVal - rCheekPercentVal) <= 5 {
                degrees.append(firstHeading + count)
            }
            count += 1
        }
        
        // TODO: Find best degree. Currently unimplemented.
        return 0
    }
}
