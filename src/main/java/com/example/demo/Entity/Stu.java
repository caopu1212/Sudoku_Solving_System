package com.example.demo.Entity;

public class Stu {

    public String sno;
    public String sname;
    public String password;
    public String tno;
    public String tname;
    public String tgrade;

    public String getSno() {
        return sno;
    }

    public void setSno(String sno) {
        this.sno = sno;
    }

    public String getSname() {
        return sname;
    }

    public void setSname(String sname) {
        this.sname = sname;
    }

    public String getPassword() {
        return password;
    }

    public void setPassword(String password) {
        this.password = password;
    }

    public String getTno() {
        return tno;
    }

    public void setTno(String tno) {
        this.tno = tno;
    }

    public String getTname() {
        return tname;
    }

    public void setTname(String tname) {
        this.tname = tname;
    }

    public String getTgrade() {
        return tgrade;
    }

    public void setTgrade(String tgrade) {
        this.tgrade = tgrade;
    }

    @Override
    public String toString() {
        return "Stu{" +
                "sno='" + sno + '\'' +
                ", sname='" + sname + '\'' +
                ", password='" + password + '\'' +
                ", tno='" + tno + '\'' +
                ", tname='" + tname + '\'' +
                ", tgrade='" + tgrade + '\'' +
                '}';
    }

}
