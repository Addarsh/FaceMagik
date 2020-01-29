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
import Alamofire

class ViewController: UIViewController, ARSCNViewDelegate {
    @IBOutlet weak var captureButton: UIButton!
    
    @IBOutlet var sceneView: ARSCNView!
    
    private let concurrentPhotoQueue = DispatchQueue(label: "com.ARFoundation.photoqueue", attributes: .concurrent)
    
    private var lastCapturedImage: CVPixelBuffer?
    
    private var pngData: Data?
    
    private let uploadURL = "http://[2601:647:4200:af70:a94d:5325:6b1c:b020]:8000/skin/"
    
    private var lastFaceAnchor: ARFaceAnchor?
    
    private var lastCamera: ARCamera?
    
    private var lastLightEstimate: ARDirectionalLightEstimate?
    
    private var lightDict: [String: [Float]]?
    
    private var vertices: [[Int]]?
    
    private var normals: [[Float]]?

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
            
            self.lastFaceAnchor = faceAnchor
            self.lastCamera = self.sceneView.session.currentFrame?.camera
            self.lastCapturedImage = self.sceneView.session.currentFrame?.capturedImage
            self.lastLightEstimate = self.sceneView.session.currentFrame?.lightEstimate as? ARDirectionalLightEstimate
        }
    }
    
    @IBAction func onClick(_ sender: Any) {
        concurrentPhotoQueue.sync {
            guard let lastImage = self.lastCapturedImage else {
                return
            }
            self.pngData =  UIImage(pixelBuffer: lastImage)?.pngData()
            self.vertices = projectedVertices()
            self.normals = calcNormals()
            self.lightDict = lightEstimate()
        }
        
        guard let pngData = self.pngData else {
            return
        }
        guard let vertices = self.vertices else {
            return
        }
        guard let lightDict = self.lightDict else {
            return
        }
        guard let normals = self.normals else {
            return
        }
        
        guard let jsonVertices = try? JSONSerialization.data(withJSONObject: vertices) else {
            return
        }
        
        guard let jsonLight = try? JSONSerialization.data(withJSONObject: lightDict) else {
            return
        }
        
        guard let jsonNormals =  try? JSONSerialization.data(withJSONObject: normals) else {
            return
        }
        
        Alamofire.upload(multipartFormData: { multipartFormData in
            multipartFormData.append(pngData.base64EncodedData(), withName: "fileset", mimeType: "image/png")
            multipartFormData.append(jsonVertices, withName: "vertices")
            multipartFormData.append(jsonLight, withName: "lighting")
            multipartFormData.append(jsonNormals, withName: "normals")
        }, to: uploadURL) { result in
            switch result {
            case .success(let upload, _, _):
                upload.uploadProgress{progress in
                    print ("Upload progress \(progress.fractionCompleted)")
                }
                
                upload.responseJSON { response in
                    print ("Upload complete")
                }
            case .failure(let encodingError):
                print (encodingError)
            }
        }
    }
    
    func lightEstimate() -> [String:[Float]]? {
        guard let lightEstimate = self.lastLightEstimate else {
            return nil
        }
        let spHarmonics = lightEstimate.sphericalHarmonicsCoefficients
        var lightDict: [String:[Float]] = [:]
        lightDict["colorTemperature"] = [Float(lightEstimate.ambientColorTemperature)]
        
        let floatSz = 4 // 4 bytes.
        let blockSz = 36 // 36 bytes for each channel.
        
        var redArr: [Float] = []
        for offset in stride(from: 0, to: blockSz, by: floatSz) {
            redArr.append(spHarmonics.withUnsafeBytes{ $0.load(fromByteOffset: offset, as: Float.self)})
        }
        lightDict["red"] = redArr
        
        var greenArr: [Float] = []
        for offset in stride(from: blockSz, to: 2*blockSz, by: floatSz) {
            greenArr.append(spHarmonics.withUnsafeBytes{ $0.load(fromByteOffset: offset, as: Float.self)})
        }
        lightDict["green"] = greenArr
        
        var blueArr: [Float] = []
        for offset in stride(from: 2*blockSz, to: 3*blockSz, by: floatSz) {
            blueArr.append(spHarmonics.withUnsafeBytes{ $0.load(fromByteOffset: offset, as: Float.self)})
        }
        lightDict["blue"] = blueArr
        return lightDict
    }
    
    func projectedVertices() -> [[Int]]? {
        guard let image = self.lastCapturedImage else {
            return nil
        }
        guard let faceAnchor = self.lastFaceAnchor else {
            return nil
        }
        guard let camera = self.lastCamera else {
            return nil
        }
        let geometry = faceAnchor.geometry
        let vertices = geometry.vertices
        
        let width = CVPixelBufferGetWidth(image)
        let height = CVPixelBufferGetHeight(image)
        
        let textureCoordinates = vertices.map { vertex -> [Int] in
            let world_vector3 = toWorldCords(vertex: vertex, faceAnchor: faceAnchor)
            let pt = camera.projectPoint(world_vector3,
                orientation: .portrait,
                viewportSize: CGSize(
                    width: CGFloat(height),
                    height: CGFloat(width)))
            return [Int(pt.y), Int(pt.x)]
        }
        return textureCoordinates
    }
    
    // toWorldCords converts given face vertex from face
    // coordinate system to world coordinates.
    func toWorldCords(vertex: vector_float3, faceAnchor: ARFaceAnchor) -> simd_float3 {
        let vertex4 = vector_float4(vertex.x, vertex.y, vertex.z, 1)
        let world_vertex4 = simd_mul(faceAnchor.transform, vertex4)
        return simd_float3(x: world_vertex4.x, y: world_vertex4.y, z: world_vertex4.z)
    }
    
    func calcNormals() -> [[Float]]? {
        guard let faceAnchor = self.lastFaceAnchor else {
            return nil
        }
        
        //let vertices = faceAnchor.geometry.vertices
        let triangleIndices = faceAnchor.geometry.triangleIndices
        let triangleCount = faceAnchor.geometry.triangleCount
        let vertices = faceAnchor.geometry.vertices
        
        var count = 0
        var neighborMap = [Int:[Set<Int>]]()
        
        // Map vertex to neighgboring faces.
        while count != triangleCount {
            let face: Set<Int> = [Int(triangleIndices[count*3]), Int(triangleIndices[count*3 + 1]), Int(triangleIndices[count*3 + 2])]
            face.forEach { index  in
                if neighborMap[index] != nil {
                    neighborMap[index]!.append(face)
                } else {
                    neighborMap[index] = [face]
                }
            }
            count += 1
        }
        
        // Calculate normal for each vertex.
        let normals = vertices.enumerated().map { (index, _) -> [Float] in
            let faces = neighborMap[index]!
            var nList: [simd_float3] = []
            for face in faces {
                var simdArr: [simd_float3] = []
                for vIdx in face {
                    let v = toWorldCords(vertex: vertices[vIdx], faceAnchor: faceAnchor)
                    simdArr.append(simd_float3(v.x, v.y, v.z))
                }
                let vector1 = simdArr[1] - simdArr[2]
                let vector2 = simdArr[1] - simdArr[0]
                nList.append(simd_normalize(simd_cross(vector1, vector2)))
            }
            var avgNormal: simd_float3 = simd_float3(0, 0, 0)
            for n in nList {
                avgNormal += n
            }
            avgNormal = simd_normalize(avgNormal/Float(nList.count))
            return [Float(avgNormal[0]), Float(avgNormal[1]), Float(avgNormal[2])]
        }
        return normals
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
        guard let cImage = cgImage else {
            return nil
        }
        self.init(cgImage: cImage)
    }
}
