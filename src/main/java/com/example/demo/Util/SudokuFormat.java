package com.example.demo.Util;

import java.util.Arrays;


/**
 * 去除多余字符
 */
public class SudokuFormat {
    public String modify(String Sudoku) {
        String dealed_sudoku;
        dealed_sudoku = Sudoku.replace("[", "").replace("]", "").replace(" ", "");
        return dealed_sudoku;
    }



    //字符串数独转二维数组数独
    public static int[][] str_to_list_sudoku(String str_sudoku) {
        //字符串转string数组
        String[] str_sudoku_stringlist = str_sudoku.split(",");
        //string数组转int数组
        int[] str_sudoku_intlist = Arrays.stream(str_sudoku_stringlist).mapToInt(Integer::parseInt).toArray();


        int[][] sudoku_temp = new int[9][9];
        int count = 0;
        for (int column = 0; column < 9; column++) {
            for (int line = 0; line < 9; line++) {
                sudoku_temp[column][line] = str_sudoku_intlist[count];
                count++;
            }
        }
        return sudoku_temp;
    }

}
