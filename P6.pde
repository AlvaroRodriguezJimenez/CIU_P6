import java.lang.*;
import processing.video.*;
import cvimage.*;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

Capture cam;
CVImage img1, img2, img3, img4, grayimg, cannyimg, threshimg, moveimg, moveaux;

void setup() {
  size(1200, 700);
  background(125);
 
  cam = new Capture(this, width, height);
  cam.start();

  System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
  println(Core.VERSION);

  img1 = new CVImage(cam.width, cam.height);
  img2 = new CVImage(cam.width, cam.height);
  img3 = new CVImage(cam.width, cam.height);
  img4 = new CVImage(cam.width, cam.height);
  
  grayimg = new CVImage(cam.width, cam.height);
  cannyimg = new CVImage(cam.width, cam.height);
  threshimg = new CVImage(cam.width, cam.height);
  moveimg = new CVImage(cam.width, cam.height);
  moveaux = new CVImage(cam.width, cam.height);
}

void draw() {  
  if (cam.available()) {
    background(0);
    cam.read();
    
    gray();
    canny();
    thresh();
    move();
    
    
  }
}


void  cpMat2CVImage(Mat in_mat,CVImage out_img){    
  byte[] data8 = new byte[cam.width*cam.height];

  out_img.loadPixels();
  in_mat.get(0, 0, data8);

  // Cada columna
  for (int x = 0; x < cam.width; x++) {
    // Cada fila
    for (int y = 0; y < cam.height; y++) {
      // Posición en el vector 1D
      int loc = x + y * cam.width;
      //Conversión del valor a unsigned
      int val = data8[loc] & 0xFF;
      //Copia a CVImage
      out_img.pixels[loc] = color(val);
    }
  }
  out_img.updatePixels();
}

void gray(){
  
  img1.copy(cam, cam.width/2, -cam.height/2, cam.width, cam.height,
    0, img1.height/2, img1.width/2, img1.height/2);
    img1.copyTo();
  
    image(img1, img1.width/2, -img1.height/2);
    
    
}

void canny(){

  img2.copy(cam, 0, 0, cam.width/2, cam.height/2,
    img2.width/2, img2.height/2, img2.width/2, img2.height/2);
    img2.copyTo();  
  
    Mat cannyaux = img2.getGrey();
    //Gradiente
    int ddepth = CvType.CV_16S;
    Mat grad_x = new Mat();
    Mat grad_y = new Mat();
    Mat abs_grad_x = new Mat();
    Mat abs_grad_y = new Mat();

    // Gradiente X
    Imgproc.Sobel(cannyaux, grad_x, ddepth, 1, 0);
    Core.convertScaleAbs(grad_x, abs_grad_x);

    // Gradiente Y
    Imgproc.Sobel(cannyaux, grad_y, ddepth, 0, 1);
    Core.convertScaleAbs(grad_y, abs_grad_y);

    // Gradiente total aproximado
    Core.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, cannyaux);
    
    cpMat2CVImage(cannyaux, cannyimg);
    image(cannyimg, -img2.width/2, -img2.height/2);
    cannyaux.release();
}

void thresh(){

  img3.copy(cam, -cam.width/2, cam.height/2, cam.width, cam.height,
    img3.width/2, 0, img3.width/2, img3.height/2);
    img3.copyTo();  
  
    Mat threshaux = img3.getGrey();
    Imgproc.threshold(threshaux, threshaux, 100, 255, Imgproc.THRESH_BINARY);
    cpMat2CVImage(threshaux, threshimg);
    
    image(threshimg, -img3.width/2, img3.height/2);
    threshaux.release();
}

void move(){

  img4.copy(cam, cam.width/2, cam.height/2, cam.width, cam.height,
    0, 0, img4.width/2, img4.height/2);
    img4.copyTo();

  Mat gris = img4.getGrey();
  Mat pgris = moveimg.getGrey();

  Core.absdiff(gris, pgris, gris);
  cpMat2CVImage(gris, moveaux);

  image(moveaux, img4.width/2, img4.height/2);
  moveimg.copy(img4, 0, 0, img4.width, img4.height, 0, 0, img4.width, img4.height);
  moveimg.copyTo();
  gris.release();
}
