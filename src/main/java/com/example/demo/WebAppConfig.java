package com.example.demo;

import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.EnableWebMvc;
import org.springframework.web.servlet.config.annotation.ResourceHandlerRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration

@EnableWebMvc
public class WebAppConfig implements WebMvcConfigurer {

//    @Value("${cbs.imagesPath}")
//    private String mImagesPath;
//
    //上传到cbs.imagesPath配置的目录中的图片，在项目中可以访问，访问是时标签的src属性的属性值是"/images/图片名.扩展名"。
//    @Override
//    public void addResourceHandlers(ResourceHandlerRegistry registry)
//    {
//        if (mImagesPath.equals("") || mImagesPath.equals("${cbs.imagesPath}"))
//        {
//            String imagesPath = WebAppConfig.class.getClassLoader().getResource("").getPath();
//            if (imagesPath.indexOf(".jar") > 0)
//            {
//                imagesPath = imagesPath.substring(0, imagesPath.indexOf(".jar"));
//            }
//            else if (imagesPath.indexOf("classes") > 0)
//            {
//                imagesPath = "file:" + imagesPath.substring(0, imagesPath.indexOf("classes"));
//            }
//            imagesPath = imagesPath.substring(0, imagesPath.lastIndexOf("/")) + "/images/";
//            mImagesPath = imagesPath;
//        }
//        LoggerFactory.getLogger(WebAppConfig.class).info("imagesPath=" + mImagesPath);
//
//
//        registry.addResourceHandler("/images/**").addResourceLocations(mImagesPath);
//
//        registry.addResourceHandler("/**").addResourceLocations(
//                "classpath:/static/");
//        registry.addResourceHandler("swagger-ui.html").addResourceLocations(
//                "classpath:/META-INF/resources/");
//        registry.addResourceHandler("/webjars/**").addResourceLocations(
//                "classpath:/META-INF/resources/webjars/");
//        WebMvcConfigurer.super.addResourceHandlers(registry);
//    }



    private static final String[] CLASSPATH_RESOURCE_LOCATIONS = {
            "classpath:/META-INF/resources/", "classpath:/resources/",
            "classpath:/static/", "classpath:/public/" };

    @Override
    public void addResourceHandlers(ResourceHandlerRegistry registry) {
        registry.addResourceHandler("/**").addResourceLocations(
                "classpath:/static/");
        registry.addResourceHandler("swagger-ui.html").addResourceLocations(
                "classpath:/META-INF/resources/");
        registry.addResourceHandler("/webjars/**").addResourceLocations(
                "classpath:/META-INF/resources/webjars/");
        WebMvcConfigurer.super.addResourceHandlers(registry);
    }

//    @Override
//    public void addResourceHandlers(ResourceHandlerRegistry registry) {
//        registry.addResourceHandler("/static/**").addResourceLocations("classpath:/static/");
//    }


}
