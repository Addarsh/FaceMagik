//
//  ImageViewController.swift
//  ARFoundation
//
//  Created by Addarsh Chandrasekar on 1/24/20.
//  Copyright © 2020 Addarsh Chandrasekar. All rights reserved.
//

import UIKit

class ImageViewController: UIViewController {
    @IBOutlet weak var imageView: UIImageView!
    
    var capturedImage: UIImage?
    
    override func viewDidLoad() {
        super.viewDidLoad()

        // Do any additional setup after loading the view.
        print ("captured image \(capturedImage!.size)")
        
        self.view.addSubview(UIImageView(image: capturedImage))
    }
    

    /*
    // MARK: - Navigation

    // In a storyboard-based application, you will often want to do a little preparation before navigation
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        // Get the new view controller using segue.destination.
        // Pass the selected object to the new view controller.
    }
    */

}
