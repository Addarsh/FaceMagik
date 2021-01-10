//
//  SkinToneAnalyzer.swift
//  FaceMagik
//
//  Created by Addarsh Chandrasekar on 1/9/21.
//

import UIKit

class SkinToneAnalyzer: AssessFaceControllerDelegate {
    private struct ImageDetails {
        var image: CIImage
        var fullFaceMask: CIImage
    }
    
    private var headingImageDetailsMap: [Int:ImageDetails] = [:]
    private let headingQueue = DispatchQueue(label: "Heading Queue to analyze skin tone", qos: .userInitiated , attributes: [], autoreleaseFrequency: .inherit, target: nil)
    private var currImageDetails: ImageDetails?
    
    init() {}
    
    func handleUpdatedHeading(heading: Int) {
        self.headingQueue.async {
            guard let imageDetails = self.currImageDetails else {
                return
            }
            if self.headingImageDetailsMap[heading] != nil {
                return
            }
            self.headingImageDetailsMap[heading] = imageDetails
        }
    }
    
    func handleUpdatedImage(image: CIImage?, fullFaceMask: CIImage?) {
        self.headingQueue.async {
            guard let img = image else {
                return
            }
            guard let fullMask = fullFaceMask else {
                return
            }
            self.currImageDetails = ImageDetails(image: img, fullFaceMask: fullMask)
        }
    }
}
