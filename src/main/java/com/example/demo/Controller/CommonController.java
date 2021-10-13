package com.example.demo.Controller;

import com.example.demo.Service.CommonService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.HttpSession;
import java.io.IOException;

//import test.service.CommonService;


@Controller
public class CommonController {
    @Autowired
    public CommonService commonservice;

    @RequestMapping(value = "/login", method = {RequestMethod.POST, RequestMethod.GET})
    public String login() {

        System.out.println("entered here");
        return "login";
    }





    @RequestMapping(value = "/loginPage", method = {RequestMethod.POST, RequestMethod.GET})
    public String login(HttpServletRequest request, HttpServletResponse response, HttpSession session) {
        String tno = request.getParameter("tno");
        String password = request.getParameter("password");
        System.out.println("你输入的用户名为：" + tno);
        System.out.println("你输入的密码为：" + password);
        String tname = commonservice.login(tno, password);
        session.setAttribute("tname", tname);
        if (tname == null) {
            //返回弹窗
            try {
                response.getWriter().write("<script>alert('账号或密码错误哦哦哦！');</script>");
            } catch (IOException e) {
                e.printStackTrace();
            }
            System.out.println("账户或密码错误");
            return "login";

        } else {
           return "confirm";
        }
    }




//    @RequestMapping(value = "/index", method = {RequestMethod.POST, RequestMethod.GET})
//    public String loginindex() {
//        return "/login/test";
//
//    }

}
