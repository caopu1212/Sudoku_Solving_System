package com.example.demo.Mapper;

import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import org.springframework.stereotype.Component;


@Component
public interface  Common  {

    @Select("select sname from stu where sno=#{sno} and password=#{password}")
    public String login(@Param("sno") String sno,@Param("password") String password);


}
