import java.io.IOException;

public class cmdTest {
    public static void main(String[] args) throws IOException {
        String path = "python D:\\作业\\练习\\src\\main\\resources\\python_script\\test.py";

        Runtime runtime=Runtime.getRuntime();
        try{
//            runtime.exec("cmd /c start "+ path);
            runtime.exec("cmd /k start "+ path);
        }catch(Exception e){
            System.out.println("Error!");
        }


    }

}
