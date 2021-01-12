//
//  SkinToneAnalyzer.swift
//  FaceMagik
//
//  Created by Addarsh Chandrasekar on 1/9/21.
//

import UIKit

class SkinToneAnalyzer: AssessFaceControllerDelegate {
    private var headingImageDetailsMap: [Int:FaceProperties] = [:]
    private let headingQueue = DispatchQueue(label: "Heading Queue to analyze skin tone", qos: .userInitiated , attributes: [], autoreleaseFrequency: .inherit, target: nil)
    private var currFaceProperties: FaceProperties?
    
    init() {}
    
    func handleUpdatedHeading(heading: Int) {
        self.headingQueue.async {
            guard let faceProperties = self.currFaceProperties else {
                return
            }
            if self.headingImageDetailsMap[heading] != nil {
                return
            }
            self.headingImageDetailsMap[heading] = faceProperties
        }
    }
    
    func handleUpdatedImage(faceProperties: FaceProperties) {
        self.headingQueue.async {
            self.currFaceProperties = faceProperties
        }
    }
}
