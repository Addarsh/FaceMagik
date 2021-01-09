//
//  BadColorTemperature.swift
//  FaceMagik
//
//  Created by Addarsh Chandrasekar on 1/6/21.
//

import UIKit

class BadColorTemperature: UIViewController {
    
    static func storyboardInstance() -> BadColorTemperature? {
        let className = String(describing: BadColorTemperature.self)
        let storyboard = UIStoryboard(name: className, bundle: nil)
        return storyboard.instantiateInitialViewController() as? BadColorTemperature
    }
    
    // tryAgain allowes user to go back to previous view controller.
    @IBAction func tryAgain() {
        self.dismiss(animated: true)
    }
}
