//
//  ViewController.swift
//  ARFoundation
//
//  Created by Addarsh Chandrasekar on 1/20/20.
//  Copyright © 2020 Addarsh Chandrasekar. All rights reserved.
//

import UIKit
import SceneKit
import ARKit
import VideoToolbox

class ViewController: UIViewController, ARSCNViewDelegate {
    @IBOutlet weak var captureButton: UIButton!
    
    @IBOutlet var sceneView: ARSCNView!
    
    private let concurrentPhotoQueue = DispatchQueue(label: "com.ARFoundation.photoqueue", attributes: .concurrent)
    
    private var lastCapturedImage: CVPixelBuffer?
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // Set the view's delegate
        sceneView.delegate = self
        
        // Show statistics such as fps and timing information
        sceneView.showsStatistics = true
        
        guard ARFaceTrackingConfiguration.isSupported else {
          fatalError("Face tracking is not supported on this device")
        }
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        
        // Create a session configuration
        let configuration = ARFaceTrackingConfiguration()
        configuration.isLightEstimationEnabled = true

        // Run the view's session
        sceneView.session.run(configuration, options: [.resetTracking, .removeExistingAnchors])
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        
        // Pause the view's session
        sceneView.session.pause()
    }

    // MARK: - ARSCNViewDelegate
    
/*
    // Override to create and configure nodes for anchors added to the view's session.
    func renderer(_ renderer: SCNSceneRenderer, nodeFor anchor: ARAnchor) -> SCNNode? {
        let node = SCNNode()
     
        return node
    }
*/
    
    func session(_ session: ARSession, didFailWithError error: Error) {
        // Present an error message to the user
        
    }
    
    func sessionWasInterrupted(_ session: ARSession) {
        // Inform the user that the session has been interrupted, for example, by presenting an overlay
        
    }
    
    func sessionInterruptionEnded(_ session: ARSession) {
        // Reset tracking and/or remove existing anchors if consistent tracking is required
        
    }
    
    func renderer(_ renderer: SCNSceneRenderer, nodeFor anchor: ARAnchor) -> SCNNode? {
        guard let device = sceneView.device else {
            return nil
        }
        let faceGeometry = ARSCNFaceGeometry(device: device)
        let node = SCNNode(geometry: faceGeometry)
        node.geometry?.firstMaterial?.fillMode = .lines
        return node
    }
    
    func renderer(_ renderer: SCNSceneRenderer, didUpdate node: SCNNode, for anchor: ARAnchor) {
        guard let faceAnchor = anchor as? ARFaceAnchor, let faceGeometry = node.geometry as? ARSCNFaceGeometry else {
            return
        }
        
        faceGeometry.update(from: faceAnchor.geometry)
        
        // Save last captured image.
        concurrentPhotoQueue.async(flags: .barrier) {
            [weak self] in
            guard let self = self else {
                return
            }
            self.lastCapturedImage = self.sceneView.session.currentFrame?.capturedImage
        }
    }
    
    @IBAction func onClick(_ sender: Any) {
        //performSegue(withIdentifier: "ShowImage", sender: self)
        print ("Clicked button")
    }
    
    // prepare is run just before segue. It captures last stored image
    // and send the data to the image view controller.
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        concurrentPhotoQueue.sync {
            guard let lastImage = self.lastCapturedImage else {
                return
            }
            let destVC: ImageViewController = segue.destination as! ImageViewController
            destVC.capturedImage =  UIImage(pixelBuffer: lastImage)
        }
    }
}

// Extension of UIImage to convert CVPixelBuffer to UIImagge object.
extension UIImage {
    public convenience init?(pixelBuffer: CVPixelBuffer) {
        var cgImage: CGImage?
        VTCreateCGImageFromCVPixelBuffer(pixelBuffer, options: nil, imageOut: &cgImage)
        if cgImage == nil {
            return nil
        }
        self.init(cgImage: cgImage!)
    }
}
