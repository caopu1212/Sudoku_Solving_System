package com.example.demo.Service;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.ServerSocket;
import java.net.Socket;

public class SocketService {


    private static final int BUFFER_SIZE = 1024 * 1024;
//        public  void main(String[] args) {
//
//            cmd_python("D:\\作业\\练习\\src\\main\\java\\com\\example\\demo\\Socket\\client.py");
//            System.out.println("received : ");
//            System.out.println(runSample());
//
//        }
    //运行指定目录的python脚本
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
        //服务端，接收数据前堵塞，接收后继续
        String str = null;
        try {

            ServerSocket ss = new ServerSocket(6002);

            System.out.println("waiting for connect");

            Socket socket = ss.accept();
//            Socket socket = new Socket("localhost", 6000);

            //接收
            InputStream is = socket.getInputStream();
            DataInputStream dis = new DataInputStream(is);

            //发送
            OutputStream os = socket.getOutputStream();
            DataOutputStream dos = new DataOutputStream(os);
            /*
            String a="HELLO";
            dos.writeBytes(a);
            */
//            byte message[] = {'H', 'E', 'L', 'L', 'O', '\0'};
//            dos.write(message, 0, 6);
//            System.out.println("send");
            char recv = 0;
            StringBuffer sb = new StringBuffer();


//            char[] data = new char[BUFFER_SIZE];
//            BufferedReader br = new BufferedReader(new InputStreamReader(socket.getInputStream(), charset));
//            int len = br.read(data);
//            String rexml = String.valueOf(data, 0, len);
//

//
            while (true) {
                recv = (char) (dis.readByte());
                // 终止指令
                if (recv == '\0') break;
//                System.out.print(recv);
                System.out.print("");
                sb.append(recv);
            }


//            recv = (char)(dis.readByte());

            System.out.println(recv);
            str = sb.toString();
            //字符串格式化
            str = str.replace("\"", "").replace("\\", "");

//            System.out.println(str);
            dis.close();
            dos.close();
            os.close();
            is.close();
            socket.close();
        } catch (Exception e) {
            e.printStackTrace();
        }


//        System.out.print("received : ");
//        System.out.println(str);

        return str;
    }

}
