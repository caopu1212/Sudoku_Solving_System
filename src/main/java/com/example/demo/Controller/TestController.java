package com.example.demo.Controller;

import com.example.demo.Service.SocketService;
import com.example.demo.Service.SolvingSudokuService;
import com.example.demo.Util.SudokuFormat;
import com.sun.deploy.util.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.servlet.ModelAndView;

import javax.servlet.ServletResponse;
import javax.servlet.http.HttpServletRequest;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;


@Controller
public class TestController {
    //储存前端返回的改好的数独字符串
    public static String g_returned_sudoku;

    private static final Logger LOGGER = LoggerFactory.getLogger(TestController.class);

//    @Autowired
//    public CommonService commonservice;

//    @Value("${cbs.imagesPath}")
//    private String theSetDir; //全局配置文件中设置的图片的路径

//    @GetMapping("/{page}")
//    public String toPate(@PathVariable("page") String page) {
//        return page;
//    }
//
//    @RequestMapping(value = "/upload", method = {RequestMethod.POST, RequestMethod.GET})
//    public String upload() {
//
//        System.out.println("entered upload");
//        return "upload";
//    }
//
//
//    @RequestMapping(value = "/fileUploadController")
//    public String fileUpload(MultipartFile filename, Model model) throws Exception {
//        String parentDirPath = theSetDir.substring(theSetDir.indexOf(':') + 1, theSetDir.length()); //通过设置的那个字符串获得存放图片的目录路径
//        String fileName = filename.getOriginalFilename();
//
//        File parentDir = new File(parentDirPath);
//        if (!parentDir.exists()) //如果那个目录不存在先创建目录
//        {
//            parentDir.mkdir();
//        }
//
//        filename.transferTo(new File(parentDirPath + "image.jpg")); //全局配置文件中配置的目录加上文件名
//
//        ChangeSize a = new ChangeSize();
//        a.scaleToMnist(parentDirPath, parentDirPath);
////        model.addAttribute("pic_name", fileName);
//        return "display";
//
//    }

    /**
     * 返回跳转页面
     * 获取返回的数独字符串
     *
     * @return
     */

    @ResponseBody
    @RequestMapping(value = "/result", method = {RequestMethod.POST, RequestMethod.GET})
    public String result(HttpServletRequest request, ServletResponse response, Model model) {
        System.out.println("received : " + request.getParameter("sudoku"));
        System.out.println("received : " + request.getParameter("id"));
//        return new ModelAndView("result");
        g_returned_sudoku = request.getParameter("sudoku");
        return "http://localhost:8080/resultPage";
    }


    /**
     * 打开页面
     * 图像识别的页面
     *
     * @param request
     * @param model
     * @return
     */
    @RequestMapping(value = "/confirmPage", method = {RequestMethod.POST, RequestMethod.GET})
    public ModelAndView confirm(HttpServletRequest request, Model model) {
        System.out.println("entered confirm page");
        SudokuFormat util = new SudokuFormat();
        //格式化 转成纯字符串
        try {
            String str_sudoku = util.modify(post());
            model.addAttribute("msg", str_sudoku);
            return new ModelAndView("confirm");
        } catch (NullPointerException e) {
            System.out.println(e);
        }

//        String str_sudoku = "0,6,1,0,3,0,0,2,0,0,5,0,0,0,8,1,0,7,0,0,0,0,0,7,0,3,4,0,0,9,0,0,6,0,7,8,0,0,3,2,0,9,5,0,0,5,7,0,3,0,0,9,0,0,1,9,0,7,0,0,0,0,0,8,0,2,4,0,0,0,6,0,0,4,0,0,1,0,2,5,0,";
        return new ModelAndView("index");

    }


    /**
     * 打开页面
     * 显示结果的页面
     *
     * @param request
     * @param model
     * @return
     */
    @RequestMapping(value = "/resultPage", method = {RequestMethod.POST, RequestMethod.GET})
    public ModelAndView result(HttpServletRequest request, Model model) {
        System.out.println("entered result page");
//        String a = "0,6,1,0,3,0,0,2,0,0,5,0,0,0,8,1,0,7,0,0,0,0,0,7,0,3,4,0,0,9,0,0,6,0,7,8,0,0,3,2,0,9,5,0,0,5,7,0,3,0,0,9,0,0,1,9,0,7,0,0,0,0,0,8,0,2,4,0,0,0,6,0,0,4,0,0,1,0,2,5,0,";
        int[][] sudoku_temp = SudokuFormat.str_to_list_sudoku(g_returned_sudoku);
//        int[][] sudoku_temp = SudokuFormat.str_to_list_sudoku(a);
        sudoku_temp = SolvingSudokuService.sudoku_put_onlyone(sudoku_temp);
        try {
            ArrayList b = SolvingSudokuService.solve(sudoku_temp, 0, 0);
            String list_str = StringUtils.join(b, ",");
            System.out.println(list_str);//a,b,c
            model.addAttribute("msg", list_str);
        } catch (Exception e) {
        }
        return new ModelAndView("result");
    }


    /**
     * 打开页面
     * 上传图片
     * 返回确认页面的地址
     */
    @RequestMapping("/uploadPage")
    public String upload() {
        return "upload";
    }

    @RequestMapping(value = "/upload", method = {RequestMethod.POST, RequestMethod.GET})
    @ResponseBody
    public ModelAndView upload(@RequestParam("file") MultipartFile file) {
        if (file.isEmpty()) {
            return new ModelAndView("failed ，please select available image");
        }
        String fileName = file.getOriginalFilename();
        String filePath = "C:\\Users\\Administrator\\Desktop\\testimage\\uploaded\\";
        File dest = new File(filePath + "image.jpg");
        try {
            if (!dest.exists()) {
                dest.createNewFile();
            }
            file.transferTo(dest);
            LOGGER.info("succeed");
            return new ModelAndView("redirect:/confirmPage");
        } catch (IOException e) {
            LOGGER.error(e.toString(), e);
        }
        return new ModelAndView("failed");
    }


    /**
     * 打开页面
     * 手动填数字的页面
     *
     * @param request
     * @param model
     * @return
     */
    @RequestMapping(value = "/confirm2Page", method = {RequestMethod.POST, RequestMethod.GET})
    public ModelAndView confirm2(HttpServletRequest request, Model model) {
        return new ModelAndView("confirm2");
    }


    @RequestMapping("/index")
    public String index() {
        return "index";
    }


    @RequestMapping("/playPage")
    public String paly() {
        return "play";
    }


    //通过socket发送给python预测程序，进行cnn计算
    public String post() {
        SocketService s = new SocketService();
        s.cmd_python("D:\\作业\\FYP\\源码\\Sudoku_solving_system\\src\\main\\java\\com\\example\\demo\\Socket\\client.py");
        String str_sudoku = s.runSample();
        return str_sudoku;
    }

    public static void main(String[] args) {


    }

}
