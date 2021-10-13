package com.example.demo.Util;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class ChangeSize {

    public void scaleToMnist(String srcImageFile,String result) {
        try
        {
            BufferedImage src = ImageIO.read(new File(srcImageFile));
            float width = src.getWidth();
            float height = src.getHeight();

            float scale = 28 / width;
            width = width * scale;
            height = height * scale;

            Image image = src.getScaledInstance((int)width, (int)height, Image.SCALE_SMOOTH);
            BufferedImage tag = new BufferedImage((int)width, (int)height, BufferedImage.TYPE_INT_RGB);
            Graphics g = tag.getGraphics();
            g.drawImage(image, 0, 0, null);
            g.dispose();
            ImageIO.write(tag, "JPEG", new File(result));
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }
    }
}
