# Driver-Drowsiness-Detection
An Open-CV based python apllication which indicate whether the Driver is sleepy or not with various levels of Alerts.

## Setting Up the Application (Ubuntu)
1. Install python and pip if not have on PC.  
  
  
    <code>sudo apt-get install python3.6 
    sudo apt install python-pip</code>  
      
      
2. Execute the following command in terminal:  
  
  
    <code>pip install -r requirements.txt</code>  
      
      
3. Then, just simply run the application  
  
  
    <code>python final-integration.py</code>  

## Setting Up Application (Windows) 
1. Install Anaconda package from there distribution.  
     
2. Install necessary packages as mentioned in <code> requirements.txt </code> file.  
   
3. Run <code> final-integration.py </code> file.  
  
## Future Works :  
1. Introduce Neck bend feature for sleeping indicator.  
   
2. Implementing above using a DNN and comparing the results.
   
3. Integrate with a web up.  
   
4. Allow autodrive or automatic stopping (not instantaneous) of vehicle in case of Dangerous/Alarm level 3.  
   



## FAQ  
1. If No Camera Detected : Try to change port no. in <code>cv2.VideoCapture(<b><u>-1</u></b>)</code> to either 0 or -1.  