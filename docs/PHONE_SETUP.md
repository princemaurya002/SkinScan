# Using your Motorola Edge 60 as the SkinScan camera

You can use your phone as the camera by streaming its video over WiFi to your PC. Follow these steps.

---

## Step 1: Install a streaming app on your Motorola Edge 60

Install one of these free Android apps from the Play Store:

- **IP Webcam** (recommended) – [Play Store](https://play.google.com/store/apps/details?id=com.pas.webcam)
- Or **DroidCam** (WiFi or USB mode)

This guide uses **IP Webcam**. Steps are similar for DroidCam (use the URL or client it provides).

---

## Step 2: Connect phone and PC to the same WiFi

- Turn on WiFi on your Motorola Edge 60.
- Connect your **PC/laptop** to the **same WiFi network** (no guest network; same subnet).
- Note: The phone will show an IP address in the app (e.g. `192.168.1.105`). Your PC must be able to reach it.

---

## Step 3: Start the camera server on the phone

1. Open **IP Webcam** on your phone.
2. Scroll down and tap **Start server** (or “Start”).
3. The app will show a URL, for example:
   - `http://192.168.1.105:8080`
   - Or with video path: `http://192.168.1.105:8080/video`
4. Leave the app in the foreground or allow it to run in the background (check app settings if the stream stops when you leave the app).
5. **Optional:** In IP Webcam settings you can set resolution (e.g. 1280x720) and disable “Front camera” if you want to use the **back camera** for better quality.

---

## Step 4: Set the URL in SkinScan

1. On your PC, open the project folder and edit **config.json**.
2. Set `camera_type` to `"phone"`.
3. Set `phone_cam_url` to the URL from the app. Use the **video** URL so OpenCV can read the stream:
   - IP Webcam: `http://192.168.1.XXX:8080/video`  
     Replace `192.168.1.XXX` with the IP shown in the app (e.g. `192.168.1.105`).
   - Port is usually `8080`; if you changed it in the app, use that port.

Example **config.json**:

```json
{
  "camera_type": "phone",
  "phone_cam_url": "http://192.168.1.105:8080/video",
  "n_gradient_images": 2,
  "capture_new_data": true
}
```

4. Save the file.

---

## Step 5: Test the connection

1. On your PC, run:

   ```bash
   python skinscan_depth_pipeline.py
   ```

2. You should see: `Using phone camera at: http://192.168.1.XXX:8080/video`
3. The capture window will show gradient patterns on the **PC screen**. Your **phone** (as the camera) must be aimed at the same scene that the screen is illuminating (e.g. your hand or a subject on a table).
4. **Position the phone** so it sees:
   - The screen (or the light from it) and  
   - The subject (e.g. skin) in the same frame.

If you get “unable to open phone camera” or “did not return a frame”:

- Check that the phone IP and port in `phone_cam_url` match the app.
- Open the URL in a **browser on the PC** (e.g. `http://192.168.1.105:8080`). You should see the IP Webcam page; then try `http://192.168.1.105:8080/video` or the video link from the app.
- Ensure no firewall on the PC is blocking the port (e.g. 8080).
- Make sure the phone is not in “sleep” and the app is still streaming.

---

## Step 6: Capture workflow with the phone

1. **Mount the phone** so it is stable (tripod, stand, or fixed position). Avoid moving it during capture.
2. **Position:** Phone camera and PC screen both facing the subject (e.g. 25–40 cm from the subject). The screen will show the gradient patterns; the phone captures the subject under those patterns.
3. Run:

   ```bash
   python skinscan_depth_pipeline.py
   ```

4. The PC will display the patterns; the phone (streaming to the PC) will take the images. Wait for all 5 frames to be captured.
5. Results appear in the **Results/** folder (depth, normals, mesh, etc.).

---

## Optional: Use back camera for better quality

For 3D scanning, the **back camera** of the Motorola Edge 60 usually gives better resolution and stability:

- In **IP Webcam**, open **Settings** (or the gear icon) and disable “Front camera” / use “Back camera”.
- Restart the server and update `phone_cam_url` in config.json if the URL changes.

---

## Troubleshooting

| Problem | What to do |
|--------|------------|
| “unable to open phone camera” | Check IP and port in config.json; open the URL in a browser on the PC; same WiFi for phone and PC. |
| “did not return a frame” | Restart the app on the phone; check firewall; try lowering resolution in the app. |
| Very slow or laggy | In the app, reduce resolution (e.g. 720p); move the PC closer to the WiFi router. |
| Stream stops when app is in background | In Android battery settings, allow IP Webcam to run in background / disable battery optimization for the app. |

---

## Summary

1. Install **IP Webcam** (or DroidCam) on your Motorola Edge 60.  
2. Connect phone and PC to the **same WiFi**.  
3. **Start server** in the app and copy the URL (e.g. `http://192.168.1.105:8080/video`).  
4. Put that URL in **config.json** as `phone_cam_url` and set `camera_type` to `"phone"`.  
5. Run **python skinscan_depth_pipeline.py** and position the phone so it sees the screen and the subject.
