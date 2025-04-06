from guicar import launch_profile_and_task
from carpitch import run_game_with_profile_task  

if __name__ == '__main__':
    profile, task = launch_profile_and_task()
    print("Запуск")

    if profile and task:
        run_game_with_profile_task(profile, task)
    else:
        print("Запуск игры отменён.")
    