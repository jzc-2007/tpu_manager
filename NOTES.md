### PLACE TO RECORD SOME INTERESTING(SILLY) MISTAKES
- ``current_window=tmux display-message -p '#S:#I'`` 这个会显示**active window**的名字而不是**current window**的名字，应该用``pane_id=$TMUX_PANE;
window_id=$(tmux display-message -p -t "$pane_id" '#S:#I')``
- acquire了lock的函数不能退出（老生长谈了，但是还是犯了qwq），否则会导致deadlock
- resume应该在stage_dir,因为现在的codebase有可能被改过
- 复杂的传参数一定要用位置参数
- 注意``"3"!=3``(读取json等文件要注意)，以及字符大小写``"PREEMPTED"!='preempted'``.
- ``f-string``的大括号里用和``string``一样的引号有时候会报错，有时候不会，尽量用不一样的
- ``KeyBoardInterrupt``不属于``Exception``的子类，所以不能用``except Exception``来捕获
- `gcloud`命令可能有自动重联机制，加上`--ssh-flag="-n"`然后Ctrl+C掉tmux window可以避免这个问题，以及`kill`调远程进程的时候也要杀死父亲进程，避免远程重启（也是之前zombie进程的原因），具体Kill不掉的原因可能是多者，可以做ablation study.