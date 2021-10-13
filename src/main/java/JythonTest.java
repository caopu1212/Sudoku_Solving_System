import java.io.BufferedReader;
import java.io.InputStreamReader;

public class JythonTest {

//    public void readFileByPython(List<String> filePaths) throws FileNotFoundException {
//        URL localSrcUrl = AbstractReadFileLine.class.getResource("");
//        String localSrcPath = localSrcUrl.getPath();
//        localSrcPath = localSrcPath.substring(1, localSrcPath.length());
//        String pythonFile = localSrcPath + "PythonFileHandle.py";
//
//        int size = filePaths.size() + 2;
//        String[] args = new String[size];
//        args[0] = "python";
//        args[1] = pythonFile;
//        for (int i = 0; i < filePaths.size(); i++) {
//            int index = i + 2;
//            args[index] = filePaths.get(i);
//        }
//        try {
//
//            System.out.println("start");
//            Process proc = Runtime.getRuntime().exec(args);
//            InputStream is = proc.getErrorStream();
//            InputStreamReader isr = new InputStreamReader(is);
//            BufferedReader br = new BufferedReader(isr);
//            String line = null;
//            System.out.println("<ERROR>");
//            while ((line = br.readLine()) != null) {
//                System.out.println(line);
//                System.out.println("</ERROR>");
//                int exitValue = proc.waitFor();
//                System.out.println("Process exitValue=" + exitValue);
//            }
//            System.out.println("end");
//        } catch (Exception e) {
//            e.printStackTrace();
//        }
//
//    }

    public static void main(String[] args) {
        try {
            System.out.println("start");
            Process pr = Runtime.getRuntime().exec("python D:\\作业\\练习\\src\\main\\resources\\python_script\\client.py");

            BufferedReader in = new BufferedReader(new InputStreamReader(
                    pr.getInputStream()));
            String line;
            while ((line = in.readLine()) != null) {
                System.out.println(line);
            }
            in.close();
            pr.waitFor();
            System.out.println("end");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


}
