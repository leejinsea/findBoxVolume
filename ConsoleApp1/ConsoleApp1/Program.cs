using System;
using System.Collections.Generic;
using OpenCvSharp;

namespace ConsoleApp1
{
    class Program
    {
        static void Main(string[] args)
        {
            Mat image = Cv2.ImRead(@"..\..\1.jpg", ImreadModes.AnyColor);
            double result = GetBoxVolume(image);
            Console.WriteLine(result);
            //Cv2.ImShow("image", result);
            //Cv2.WaitKey(0);
            //Cv2.DestroyAllWindows();

        }

        static double GetBoxVolume(Mat dst)
        {
            // canny filter >> if you get caliveration image named 'img', you can get box volume. LOL!
            Mat img = dst;
            Mat blur = new Mat();
            Mat canny = new Mat();
            Mat close = new Mat();

            Point[][] contours;
            HierarchyIndex[] hierarchy;

            // First, change img use to canny
            Cv2.GaussianBlur(img, blur, new Size(3, 3), 1, 0, BorderTypes.Default);
            Cv2.Canny(blur, canny, 0, 50, 3, true);

            // Second, delete dot in your canny img with morphology
            Mat kernel = Cv2.GetStructuringElement(MorphShapes.Cross, new Size(7, 7));
            Cv2.MorphologyEx(canny, close, MorphTypes.Close, kernel, iterations: 1);

            // Third, find box most outer contours and get box volume. end!
            Cv2.FindContours(close, out contours, out hierarchy, RetrievalModes.External, ContourApproximationModes.ApproxSimple);

            double outer_area = 0;

            foreach (Point[] p in contours)
            {
                double area = -Cv2.ContourArea(p, true);
                if (outer_area < area)
                {
                    outer_area = area;
                }
            }

            int full_width = close.Size().Height * close.Size().Width;
            double box_size = outer_area / full_width;

            return box_size;

        }

    }
}
