//
//  AlertViewController.swift
//  FaceMagik
//
//  Created by Addarsh Chandrasekar on 12/2/20.
//

import UIKit

class AlertViewController: UIViewController {
    
    static func storyboardInstance() -> AlertViewController? {
        let className = String(describing: AlertViewController.self)
        let storyboard = UIStoryboard(name: className, bundle: nil)
        return storyboard.instantiateInitialViewController() as? AlertViewController
    }
}
