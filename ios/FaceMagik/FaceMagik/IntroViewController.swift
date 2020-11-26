//
//  IntroViewController.swift
//  FaceMagik
//
//  Created by Addarsh Chandrasekar on 11/26/20.
//

import UIKit


class IntroViewController: UIViewController {
    private let segueIdentifier = "envView"
    
    override func viewDidLoad() {
        super.viewDidLoad()
    }
    
    // startEnvProcessing calls Environemnt view controller.
    @IBAction func startEnvProcessing() {
        performSegue(withIdentifier: self.segueIdentifier, sender: nil)
    }
}
