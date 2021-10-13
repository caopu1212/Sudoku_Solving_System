package com.example.demo.Service;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.ServerSocket;
import java.net.Socket;

public class SocketService2 {


    private static final int BUFFER_SIZE = 1024 * 1024;

    public void cmd_python(String path) {
        Runtime runtime = Runtime.getRuntime();
        try {
            runtime.exec("cmd /k start python " + path);
//            runtime.exec("cmd /c start python " + path);
        } catch (Exception e) {
            System.out.println("Error!");
        }

    }

    public String runSample() {

        String str = null;
        try {
            ServerSocket ss = new ServerSocket(6002);
            System.out.println("waiting for connect");
            Socket socket = ss.accept();
            InputStream is = socket.getInputStream();
            DataInputStream dis = new DataInputStream(is);
            OutputStream os = socket.getOutputStream();
            DataOutputStream dos = new DataOutputStream(os);
            char recv = 0;
            StringBuffer sb = new StringBuffer();
            while (true) {
                recv = (char) (dis.readByte());
                if (recv == '\0') break;
                System.out.print("");
                sb.append(recv);
            }
            System.out.println(recv);
            str = sb.toString();
            str = str.replace("\"", "").replace("\\", "");
            dis.close();
            dos.close();
            os.close();
            is.close();
            socket.close();
        } catch (Exception e) {
            e.printStackTrace();
        }

        return str;
    }

}
